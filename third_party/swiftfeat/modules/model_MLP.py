import torch
import torch.nn as nn
import torch.nn.functional as F


from timm.models.layers import DropPath


class BasicLayer(nn.Module):
    """
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class BasicLayer_star_op(nn.Module):
    """
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 use_star_op=False):
        super().__init__()

        ################# USE STAR OPERATION #################
        self.use_star_op = use_star_op  # 是否启用星操作
        if use_star_op:
            self.f1 = nn.Conv2d(out_channels, out_channels, 1)  # 1x1卷积生成x1
            self.f2 = nn.Conv2d(out_channels, out_channels, 1)  # 1x1卷积生成x2
            self.act = nn.ReLU6()  # 与Block保持一致的激活函数
        ################# USE STAR OPERATION END #################

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
        # 星操作：增强通道交互（仅当use_star_op=True时执行）
        if self.use_star_op:
            x1 = self.act(self.f1(x))  # 激活后分支
            x2 = self.f2(x)  # 原始分支
            x = x1 * x2
        return x
    

##################    STARNET  ##################
# input: H * W * C
# output: H * W * C
# 不改变输入图像的尺寸，也不改变通道数，仅通过通道变换和非线性交互提升特征表示能力，适用于需要保持高分辨率的场景。

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)  # 初始化BN层的权重为1
            torch.nn.init.constant_(self.bn.bias, 0)  # 初始化BN层的偏置为0


class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


############### GateMLP Begain ###############   
class GatedMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2, affine=False),
            nn.GLU(),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2, affine=False),
            nn.GLU(),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2, affine=False),
            nn.GLU(),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2, affine=False),
            nn.GLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
############### GateMLP END ###############    

    
############### S2MLP Begain ###############
class S2MLPLayer(nn.Module):
    def __init__(self, dim, hidden_dim, shift_size=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.channel_mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def spatial_shift(self, x):
        # 输入 x: (N, C)
        B, C = x.shape
        x = x.view(B, C // self.shift_size, self.shift_size)
        x = torch.roll(x, shifts=1, dims=-1)  # 模拟空间 shift
        return x.view(B, C)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.spatial_shift(x)
        x = self.channel_mlp1(x)
        return x + residual  # 残差连接
############### S2MLP END ###############


############### gMLP Begain ###############
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim // 2, dim // 2)

    def forward(self, x):
        # x: [N, dim]
        x = self.norm(x)
        u, v = x.chunk(2, dim=-1)  # split into two halves
        v = self.proj(v)
        return u * v


class gMLPLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.ReLU(inplace=True)
        self.sgu = SpatialGatingUnit(dim * 2)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.sgu(x)
        x = self.fc2(x)
        return x + residual


class FineMatcherGMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=64, depth=4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.norm_in = nn.BatchNorm1d(hidden_dim, affine=False)
        self.act_in = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(*[gMLPLayer(hidden_dim) for _ in range(depth)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [N, 128]
        x = self.fc_in(x)                    # [N, 512]
        x = self.norm_in(x)
        x = self.act_in(x)
        x = self.blocks(x)                   # [N, 512]
        x = self.fc_out(x)                   # [N, 64]
        return x
############### gMLP END ###############


class XFeatModel(nn.Module):
    """
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        """ nn.AvgPool2d(4, stride = 4) --> H/4 * W/4 """
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4),
                                   nn.Conv2d(1, 24, 1, stride=1, padding=0))

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),         # stage1-1   H*W
            BasicLayer(4, 8, stride=2),         # stage2-1   H/2 * W/2
            BasicLayer(8, 8, stride=1),         # stage2-2
            BasicLayer(8, 24, stride=2),        # stage3-1   H/4 * W/4
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),       # stage3-2
            BasicLayer(24, 24, stride=1),       # stage3-3
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),       # stage4-1   H/8 * W/8
            BasicLayer(64, 64, stride=1),       # stage4-2
            BasicLayer(64, 64, 1, padding=0),   # stage4-3
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),       # stage5-1   H/16 * W/16
            BasicLayer(64, 64, stride=1),       # stage5-2
            BasicLayer(64, 64, stride=1),       # stage5-3
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),      # stage6-1   H/32 * W/32
            BasicLayer(128, 128, stride=1),     # stage6-2
            BasicLayer(128, 128, stride=1),     # stage6-3
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),       # stage7-1   H/8 * W/8
            BasicLayer(64, 64, stride=1),       # stage7-2
            nn.Conv2d(64, 64, 1, padding=0)     # stage7-3
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        # TODO: 提升感知机性能，结合交叉注意力机制（AAAI 2024｜ETH轻量化Transformer最新研究，浅层MLP完全替换注意力模块提升性能）
        # TODO: RELU TO GELU?
        ########### ⬇️ Fine Matcher MLP ⬇️ ###########
        # 一个多层感知机（MLP），用于细粒度特征匹配。
        # 在 XFeat 模型中，该模块负责处理拼接后的局部特征描述子，也就是输入是 (N, 128) 的张量，其中 N 是匹配的特征点对数。
        # 以预测它们之间的匹配关系。
    
        # MLP
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),

            nn.Linear(512, 64),
        )

        # GateMLP512
        # self.fine_matcher = GatedMLP(128, 512, 64)

        # S2MLP
        # self.fine_matcher = nn.Sequential(
        
        #     nn.Linear(128, 512),  # feats_a: 64 + feats_b: 64 = 128
        #     nn.BatchNorm1d(512, affine=False),
        #     nn.ReLU(inplace=True),
        
        #     S2MLPLayer(dim=512, hidden_dim=512),
        #     S2MLPLayer(dim=512, hidden_dim=512),
        #     S2MLPLayer(dim=512, hidden_dim=512),
        
        #     nn.Linear(512, 64),
        # )

        # gMLP
        # self.fine_matcher = FineMatcherGMLP(
        #     input_dim=128,
        #     hidden_dim=512,
        #     output_dim=64,
        #     depth=4
        # )


    def _unfold2d(self, x, ws=2):
        """
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws) \
            .reshape(B, C, H // ws, W // ws, ws ** 2)  # B, C, H, W -> B, C, H/ws, W/ws, ws**2
        # B, C, H/ws, W/ws, ws**2 -> B, C, ws**2, H/ws, W/ws -> B, C * ws**2, H/ws, W/ws
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        """
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
        # dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # pyramid fusion
        # 使用双线性插值算法进行上采样，目标尺寸，即x3的H（x3.shape[-2]）和W（x3.shape[-1]）
        # x -> torch.Tensor(B, C, H, W): grayscale or rgb image，x3.shape = (B, C, H, W)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)  # Descriptors, Shape: (B, 64, H/8, W/8)
        # 每个描述子是一个64维向量，用于描述该位置的特征。

        # Descriptor head
        heatmap = self.heatmap_head(feats)  # Reliability map, Shape: (B, 1, H/8, W/8)
        # 用于表示每个8x8区域内特征点的可靠性分数，而不是每个像素点的可靠性分数。

        # Keypoints head，注意输入是x
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))  # Keypoint map logits, Shape:(B, 65, H/8, W/8)
        # 关键点logit图是一个二维特征图，其每个像素值表示该位置成为关键点的“原始分数”（logit）。
        # Logit图是未经过激活函数（如softmax）处理的原始输出值，通常表示某个类别的未归一化得分。

        return feats, keypoints, heatmap


################### Test Parameters and FPS ###################
if __name__ == '__main__':
    import time

    model = XFeatModel().cuda()
    model.eval()  # 设置模型为评估模式
    num_iterations = 100  # 进行多次推理以获得更稳定的平均时间
    total_time = 0.0
    with torch.no_grad():  # 关闭梯度计算以加速推理
        for _ in range(num_iterations):
            x = torch.randn(1, 3, 480, 640).cuda()
            start_time = time.time()
            y = model(x)
            end_time = time.time()
            total_time += (end_time - start_time)

    average_inference_time = total_time / num_iterations
    fps = 1 / average_inference_time
    from torchstat import stat
    from torchinfo import summary

    model1 = XFeatModel()
    # print(model1)
    summary(model1, (1, 3, 480, 640), device="cpu")  # 形状为 C，H ，W
    print(stat(model1, (3, 480, 640)))
    print(f"Average inference time: {average_inference_time * 1000:.4f} ms")
    print(f"FPS: {fps:.2f}")
