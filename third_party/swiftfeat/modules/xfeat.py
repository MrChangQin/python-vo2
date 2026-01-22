import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d


# Original XFeat weights
# myweights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt'
# swiftfeatv1
# myweights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat_default_116000_swiftfeatv1.pt'
# swiftfeatv2
# myweights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat_default_116000_swiftfeatv2.pt'
# swiftfeatv3
# myweights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat_default_169000_swiftfeatv3.pt'
# swiftfeatv5
myweights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat_default_169500_swiftfeatv5.pt'
# myweights = os.path.abspath(os.path.dirname(__file__)) + '/../SwiftFeat_checkpoints/Ablation_No_StarBlock/xfeat_default_160000.pt'


class XFeat(nn.Module):
    """
		Implements the inference module for XFeat.
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

    def __init__(self, weights = myweights, top_k=4096,
                 detection_threshold=0.05):
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
        # self.dev = torch.device('cuda')  # 选择设备  ！！！！！！！！！！！！！！！！
        # self.dev = torch.device('cpu')  # 选择设备  ！！！！！！！！！！！！！！！！
        print('Using device:', self.dev)

        self.net = XFeatModel().to(self.dev).eval()  # 加载模型并设置为评估模式
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        #  load weights if provided
        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev))
            else:
                self.net.load_state_dict(weights)

        self.interpolator = InterpolateSparse2d('bicubic')  # 双三次插值器，用于将稀疏特征图插值到稠密特征图

        # Try to import LightGlue from Kornia
        self.kornia_available = False
        self.lighterglue = None
        try:
            import kornia
            self.kornia_available = True
        except:
            pass

    #  由模型推理得到的Descriptors、Keypoint map logits 和 Reliability map计算得到特征点坐标、置信度得分和64维的描述子
    @torch.inference_mode()  # PyTorch 提供的上下文管理器装饰器，关闭梯度计算，加速推理。
    def detectAndCompute(self, x, top_k=None, detection_threshold=None):
        """
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
		"""
        if top_k is None: top_k = self.top_k  # 最大特征点数量
        if detection_threshold is None: detection_threshold = self.detection_threshold  # 检测阈值
        x, rh1, rw1 = self.preprocess_tensor(x)  # # 对输入图像进行预处理，保证图像尺寸能被32整除，避免混叠伪影

        B, _, _H1, _W1 = x.shape  # 获取输入图像的batch大小、通道数、高度和宽度

        M1, K1, H1 = self.net(x)  # 前向传播，获取Descriptors、Keypoint map logits 和 Reliability map
        M1 = F.normalize(M1, dim=1)  # 对特征进行L2归一化

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)  # 将Keypoint map logits转换为原图大小的热图，用于检测关键点
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)  # 得到NMS后的目标关键点

        # Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')  # 最近邻插值，用于计算关键点的可靠性分数
        _bilinear = InterpolateSparse2d('bilinear')  # 双线性插值，用于计算关键点的可靠性分数
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(
            -1)  # 利用 Keypoint heatmap 和 Reliability map 计算关键点的可靠性分数
        scores[torch.all(mkpts == 0, dim=-1)] = -1  # 找出所有(x=0, y=0)的无效点。将这些无效点对应的 scores 设置为 -1。

        # Select top-k features
        idxs = torch.argsort(-scores)  # 对 scores 进行降序排序，得到每个关键点的索引。
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:,
                  :top_k]  # 从 idxs 中选择前 top_k 个索引，然后使用 gather 函数从 mkpts 中提取对应的 x 坐标。
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:,
                  :top_k]  # 从 idxs 中选择前 top_k 个索引，然后使用 gather 函数从 mkpts 中提取对应的 y 坐标。
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)  # 将提取的 x 和 y 坐标拼接成一个新的张量，形状为 (B, top_k, 2)。
        scores = torch.gather(scores, -1, idxs)[:, :top_k]  # 从 idxs 中选择前 top_k 个索引，然后使用 gather 函数从 scores 中提取对应的分数。

        # Interpolate descriptors at kpts positions
        # 在低分辨率的特征图上找到高分辨率坐标对应的特征值
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)  # 利用双线性插值，从Descriptors 中提取目标关键点的特征描述子。

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)  # 对特征描述子进行 L2 归一化。范围在[0,1]之间，方便后续计算相似度。

        # Correct kpt scale
        # 调整关键点的坐标，使其与输入图像的原始尺寸相匹配。之前为了保证图像尺寸能被32整除，避免混叠伪影，现在需要将关键点的坐标调整回原始图像的尺寸。
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0  # 筛选出 scores 大于 0 的关键点。
        return [
            {'keypoints': mkpts[b][valid[b]],  # shape: (N_valid, 2)
             'scores': scores[b][valid[b]],  # shape: (N_valid,)
             'descriptors': feats[b][valid[b]]} for b in range(B)  # shape: (N_valid, 64)
            # 返回一个字典列表，每个字典包含一个图像的特征信息。
        ]

    # output[0] = {
    # 	'keypoints': torch.tensor([[100.3, 200.5],
    # 							   [300.7, 400.2],
    # 							   ...]),
    # 	'scores': torch.tensor([0.95, 0.88, ...]),
    # 	'descriptors': torch.randn(128, 64)  128表示有128个特征点，64表示每个特征点的描述子长度为64
    # }

    #  由输入得到特征点的
    @torch.inference_mode()
    def detectAndComputeDense(self, x, top_k=None, multiscale=True):
        """
			Compute dense *and coarse* descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return: features sorted by their reliability score -- from most to least
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
					'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
					'scales'       ->   torch.Tensor(top_k,): extraction scale
		"""
        if top_k is None: top_k = self.top_k
        if multiscale:  # 如果启用多尺度模式 (multiscale=True)，
            mkpts, sc, feats = self.extract_dualscale(x, top_k)  # 调用extract_dualscale方法进行多尺度特征提取，返回关键点、尺度和描述子。
        else:
            mkpts, feats = self.extractDense(x, top_k)  # 调用extractDense方法进行单尺度特征提取，并将尺度设为全1（表示没有缩放）。
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)  # 并将尺度设为全1（表示没有缩放）。

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc}  # 这种多尺度策略有助于增强模型对不同尺度目标的鲁棒性。

    # scales数值 = 1.0：表示该关键点是从原始尺寸图像中提取的。
    # scales数值 < 1.0（如0.6）：表示图像被缩小了（分辨率更低），该关键点是在低分辨率图像上检测到的。
    # scales数值 > 1.0（如1.3）：表示图像被放大了（分辨率更高），该关键点是在高分辨率图像上检测到的。

    #  使用LightGlue进行特征匹配
    @torch.inference_mode()
    def match_lighterglue(self, d0, d1, min_conf=0.1):
        """
			Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
			input:
				d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
			output:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
                                idx              -> np.ndarray (N,2) the indices of the matching features
		"""
        if not self.kornia_available:  # 检查是否安装了kornia库，如果没有安装则抛出异常。
            raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')
        elif self.lighterglue is None:  # 如果self.lighterglue为None，则导入并初始化LighterGlue模型。
            from modules.lighterglue import LighterGlue
            self.lighterglue = LighterGlue()  # 初始化LighterGlue模型。

        data = {  # 准备输入数据，包括两个图像的特征信息和图像尺寸。
            'keypoints0': d0['keypoints'][None, ...],
            'keypoints1': d1['keypoints'][None, ...],
            'descriptors0': d0['descriptors'][None, ...],
            'descriptors1': d1['descriptors'][None, ...],
            'image_size0': torch.tensor(d0['image_size']).to(self.dev)[None, ...],
            'image_size1': torch.tensor(d1['image_size']).to(self.dev)[None, ...]
        }

        # Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N]
        # matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
        out = self.lighterglue(data, min_conf=min_conf)  # 调用LighterGlue模型进行特征匹配。

        idxs = out['matches'][0]  # 提取匹配的索引。

        return d0['keypoints'][idxs[:, 0]].cpu().numpy(), d1['keypoints'][idxs[:, 1]].cpu().numpy(), out['matches'][
            0].cpu().numpy()  # 返回匹配的关键点坐标。

    # 稀疏特征匹配 sparse setting
    #  对两张输入图像进行特征提取，并通过互为最近邻（Mutual Nearest Neighbor）策略匹配它们之间的特征点。
    @torch.inference_mode()
    def match_xfeat(self, img1, img2, top_k=None, min_cossim=-1):  # min_cossim = -1 余弦相似度阈值，用于筛选有效匹配。
        """
			Simple extractor and MNN matcher.
			For simplicity it does not support batched mode due to possibly different number of kpts.
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		"""
        if top_k is None: top_k = self.top_k
        img1 = self.parse_input(img1)  # 将输入图像转换为 PyTorch 张量，并将其移动到指定的设备上（通常是 GPU）。
        img2 = self.parse_input(img2)  # 将输入图像转换为 PyTorch 张量，并将其移动到指定的设备上（通常是 GPU）。

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]  # 调用 detectAndCompute 方法提取 img1 的特征点和描述子。
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]  # 调用 detectAndCompute 方法提取 img2 的特征点和描述子。
        # out1, out2 = [
        # 	   {'keypoints': mkpts[b],  # shape: (N_valid, 2)
        #		'scores': scores[b],  # shape: (N_valid,)
        #		'descriptors': feats[b]} for b in range(B) # shape: (N_valid, 64)
        # 	# 输出是一个字典列表，每个字典包含一个图像的特征信息。
        # ]
        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)  # 调用 match 方法进行特征匹配。
        # match函数接收两个描述子张量和余弦相似度阈值作为输入，并返回匹配的特征点索引。
        # 返回的是两个列表，分别表示匹配的特征点在 img1 和 img2 中的索引。

        return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()
        # 根据匹配的索引到keypoints中找到对应索引的坐标，返回匹配的特征点坐标。

    #  半稠密特征匹配 semi-dense setting
    #  对两组图像集合进行批量特征提取、粗匹配和精化匹配，最终返回每对图像之间的精细化匹配点坐标。
    #  粗匹配使用MNN，细匹配使用MLP
    @torch.inference_mode()
    def match_xfeat_star(self, im_set1, im_set2, top_k=None):
        """
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""
        if top_k is None: top_k = self.top_k
        im_set1 = self.parse_input(im_set1)  # 将输入图像转换为 PyTorch 张量，并将其移动到指定的设备上（通常是 GPU）。
        im_set2 = self.parse_input(im_set2)  # 将输入图像转换为 PyTorch 张量，并将其移动到指定的设备上（通常是 GPU）。

        # Compute coarse feats
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)  # 调用 detectAndComputeDense 方法提取 im_set1 的特征点和描述子。
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)  # 调用 detectAndComputeDense 方法提取 im_set2 的特征点和描述子。
        # 'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
        # 'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        # 'scales'       ->   torch.Tensor(top_k,): extraction scale

        # Match batches of pairs, batch_match函数接收两个描述子张量作为输入，并返回匹配的特征点索引。
        idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'])  # 调用 batch_match 方法进行特征匹配。
        # List[Tuple[torch.Tensor, torch.Tensor]]

        B = len(im_set1)  # 获取输入图像集合的大小（batch size）。

        # Refine coarse matches
        # this part is harder to batch, currently iterate
        matches = []  # 用于存储匹配结果的列表。
        for b in range(B):
            matches.append(self.refine_matches(out1, out2, matches=idxs_list, batch_idx=b))
            # 调用 refine_matches 方法对匹配结果进行精细化匹配。
            # refine_matches函数接收两个字典和匹配的索引作为输入，并返回精细化匹配的结果。

        return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())
        # 根据匹配的索引到keypoints中找到对应索引的坐标，返回匹配的特征点坐标。


    #  将输入图像张量调整为尺寸能被32整除的形式，以避免特征提取时产生混叠伪影，并统一输入格式和设备。
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:  # 如果数组形状是 (H, W, C)
                x = torch.tensor(x).permute(2, 0, 1)[None]  # 转换为 (C,H,W)，并将其转换为 PyTorch 张量
            elif len(x.shape) == 2:  # 如果数组形状是 (H, W)（灰度图）
                x = torch.tensor(x[..., None]).permute(2, 0, 1)[None]
            else:
                raise RuntimeError('For numpy arrays, only (H,W) or (H,W,C) format is supported.')

        if len(x.shape) != 4:  # 如果张量形状不是 (B,C,H,W)
            raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')

        x = x.to(self.dev).float()  # 将张量转换为浮点数类型，并将其移动到指定的设备（self.dev）上

        H, W = x.shape[-2:]  # 获取输入张量的高度和宽度
        _H, _W = (H // 32) * 32, (W // 32) * 32  # 计算最接近的能被 32 整除的高度和宽度
        rh, rw = H / _H, W / _W  # 计算原始高度和宽度与新高度和宽度的比例因子

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)  # 使用双线性插值将输入张量的大小调整为 (_H, _W)
        return x, rh, rw

    # 用于将 Keypoint map logits 恢复为原图大小的热图（heatmap），用于表示关键点的位置信息。
    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]  # 取前64个通道，进行softmax
        B, _, H, W = scores.shape  # 形状为 B, 64, H, W

        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)  # 形状为 B, H, W, 8, 8
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap  # 形状为 B, 1, H*8, W*8，就是变成了一个原图大小的heatmap

    # 函数 NMS 实现了非极大值抑制（Non-Maximum Suppression, NMS），通过以下两个条件筛选出“高质量”的关键点：
    # 1. 局部最大值：该点在周围 kernel_size × kernel_size 窗口内是最大值（避免重复检测）
    # 2. 高于阈值：响应值必须大于给定的 threshold（过滤掉低质量 / 不显著的关键点）
    def NMS(self, x, threshold=0.05, kernel_size=5):
        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
        # Pad kpts and build (B, N, 2) tensor (N: 最多关键点数量（其余用零填充）。2: 表示 (x, y) 坐标)
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos

    #  实现了批量特征匹配，通过计算两组特征之间的余弦相似度并应用互为最近邻策略，筛选出高质量的匹配对。
    #  适用于图像对集合之间的稠密或半稠密特征匹配任务。
    @torch.inference_mode()
    def batch_match(self, feats1, feats2, min_cossim=-1):  # min_cossim=-1 表示不使用余弦相似度阈值进行匹配
        B = len(feats1)  # 获取输入特征张量的批次数（batch size）。
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))  # 计算每个 batch 中 feats1 和 feats2 的余弦相似度矩阵
        match12 = torch.argmax(cossim, dim=-1)  # 找到每个 batch 中 feats1 中的每个特征点与 feats2 中最相似的特征点的索引。
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)  # 找到每个 batch 中 feats2 中的每个特征点与 feats1 中最相似的特征点的索引。

        idx0 = torch.arange(len(match12[0]), device=match12.device)  # 生成一个从 0 到 len(match12[0])-1 的索引张量。

        batched_matches = []  # 存储匹配结果的列表。

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0  # 判断双向匹配是否一致：确保 feats1[b][i] 匹配到 feats2[b][j]，
                                                     # 同时 feats2[b][j] 也匹配回 feats1[b][i]，保证互为最近邻。

            if min_cossim > 0:  # 如果设置了余弦相似度阈值，则进一步筛选满足条件的匹配。
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]  # 结合“互为最近邻”和“相似度高于阈值”的条件，保留最终的有效匹配索引。
            else:  # 如果没有设置余弦相似度阈值，则只保留互为最近邻的匹配。
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches  # 返回匹配结果的列表，每个元素是一个包含两个张量的元组，分别表示匹配的索引。
        # 外层是一个 List，长度等于 B，即批次大小。
        # 每个元素是一个 Tuple：
        # Tuple[0]: idx0_b，形状为 (N,)，表示在当前 batch 中来自 feats1 的匹配索引。
        # Tuple[1]: idx1_b，形状为 (N,)，表示来自 feats2 的对应匹配索引。

    #  用于从 MLP 输出的 8x8 偏移分布中提取精确的偏移坐标
    def subpix_softmax2d(self, heatmaps, temp=3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(W, device=heatmaps.device), torch.arange(H, device=heatmaps.device),
                              indexing='xy')
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
        coords = coords.sum(1)

        return coords

    #  在batch_match的粗匹配后，实现了精细特征匹配，通过MLP实现
    def refine_matches(self, d0, d1, matches, batch_idx, fine_conf=0.25):  # fine_conf=0.25 表示置信度阈值，用于筛选匹配对。
        idx0, idx1 = matches[batch_idx]  # 从粗匹配结果中提取当前 batch 的匹配索引（目前是第几个batch）。
        feats1 = d0['descriptors'][batch_idx][idx0]  # 从 d0 中根据匹配索引，从描述子中提取对应的特征向量
        feats2 = d1['descriptors'][batch_idx][idx1]  # 从 d1 中根据匹配索引，从描述子中提取对应的特征向量
        mkpts_0 = d0['keypoints'][batch_idx][idx0]   # 从 d0 中提取当前 batch 的特征点坐标
        mkpts_1 = d1['keypoints'][batch_idx][idx1]   # 从 d1 中提取当前 batch 的特征点坐标
        sc0 = d0['scales'][batch_idx][idx0]  # 从 d0 中提取当前 batch 的特征点的尺度因子，用于后续坐标精调。

        # Compute fine offsets
        # 作用：将两个描述子拼接后输入到 MLP 中，预测每个匹配点的偏移量（即更精确的坐标调整）。
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))  # 调用 fine_matcher 方法进行精细匹配。
        # input: 两个描述子拼接
        # tensor([
        #     [0.1, 0.2, 0.3, 0.7, 0.8, 0.9...], 128维
        #     [0.4, 0.5, 0.6, 1.0, 1.1, 1.2...],
        #     ......
        # ])  # shape: (N, 64 + 64)
        # output：offsets 表示在 8x8 窗口内的亚像素级偏移分布

        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]  # 计算每个偏移的置信度（最大 softmax 值），用于判断该偏移是否可靠
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))  # 将偏移分布转换为实际的亚像素偏移坐标。

        mkpts_0 += offsets * (sc0[:, None])  # *0.9 #* (sc0[:,None])  # 对粗匹配的特征点坐标进行精调，调整为更精确的坐标。
        # 将偏移量乘以尺度因子后加到原始坐标上，得到更精确的特征点位置。

        mask_good = conf > fine_conf  # 构建掩码，筛选出置信度高于 fine_conf 的匹配点。它是一个布尔张量（True/False），形状与 conf 相同。
        mkpts_0 = mkpts_0[mask_good]  # 仅保留高质量的匹配点。
        mkpts_1 = mkpts_1[mask_good]  # 仅保留高质量的匹配点。

        return torch.cat([mkpts_0, mkpts_1], dim=-1)  # 将精调后的匹配点坐标拼接在一起，返回最终的匹配结果。
        # 返回值是一个二维张量，每一行表示一个匹配点对，格式为 [x1, y1, x2, y2]

    #  用于匹配两个特征集合中的特征点，通过计算特征点之间的余弦相似度来确定匹配关系。
    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim=0.82):   # 0.82 余弦相似度阈值，用于筛选有效匹配。

        cossim = feats1 @ feats2.t()  # 计算 feats1 和 feats2 的余弦相似度矩阵，即每个特征点在两组描述子之间的相似度。
        cossim_t = feats2 @ feats1.t()  # 计算转置后的余弦相似度矩阵，用于反向验证匹配是否是相互最近邻。

        _, match12 = cossim.max(dim=1)  # 找到每个特征点在 feats2 中的最大相似度对应的索引，即每个特征点在 feats2 中的匹配点。
        _, match21 = cossim_t.max(dim=1)  # 找到每个特征点在 feats1 中的最大相似度对应的索引，即每个特征点在 feats1 中的匹配点。

        idx0 = torch.arange(len(match12), device=match12.device)  # 生成一个索引张量 idx0，用于后续筛选匹配点。
        mutual = match21[match12] == idx0  # 判断双向匹配是否一致，即确保 feats1[i] 匹配到 feats2[j]，
                                           # 同时 feats2[j] 也匹配回 feats1[i]，保证互为最近邻。

        if min_cossim > 0:  # 如果指定了最小余弦相似度阈值 min_cossim，
            cossim, _ = cossim.max(dim=1)  # 计算每个特征点在 feats1 中的最大相似度，即每个特征点在 feats1 中的匹配点的相似度。
            good = cossim > min_cossim  # 筛选出相似度大于阈值的特征点。
            idx0 = idx0[mutual & good]  # 最终得到的 idx0 是满足双向匹配且相似度大于阈值的特征点的索引。
            idx1 = match12[mutual & good]  # 最终得到的 idx1 是满足双向匹配且相似度大于阈值的特征点在 feats2 中的匹配点的索引。
        else:  # 如果没有设置 min_cossim，则只保留互为最近邻的匹配。
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1  # 返回匹配的特征点索引。
        # idx0 = torch.tensor([0, 2, 5])
        # idx1 = torch.tensor([3, 1, 4]) 则意味着：
        # feats1[0] 与 feats2[3] 匹配
        # feats1[2] 与 feats2[1] 匹配
        # feats1[5] 与 feats2[4] 匹配

    #  是生成一个二维网格坐标矩阵，表示图像中所有像素点的 (x, y) 坐标。常用于特征图坐标的生成或映射。
    def create_xy(self, h, w, dev):
        # 使用 torch.meshgrid 创建二维网格坐标。
        # indexing = 'ij' 表示使用矩阵索引风格（i: 行，j: 列），即：
        # y 的形状为(h, w)，每一行是[0, 1, 2, ..., h - 1]
        # x 的形状为(h, w)，每一列是[0, 1, 2, ..., w - 1]
        y, x = torch.meshgrid(torch.arange(h, device=dev),
                              torch.arange(w, device=dev), indexing='ij')
        # 将 x 和 y 拼接成一个 (h, w, 2) 的张量，其中最后一维是 (x, y)，
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)
        return xy

    # x = [[0, 1, 2],  y = [[0, 0, 0],
    # 	 [0, 1, 2]]       [1, 1, 1]]
    # xy = [[0, 0],[1, 0],[2, 0],[0, 1],[1, 1],[2, 1]]

    #  用于提取稠密特征点和描述子，支持批量处理。通过网络输出的Reliability Map选择置信度最高的关键点，并提取对应的描述子
    def extractDense(self, x, top_k=8_000):
        if top_k < 1:
            top_k = 100_000_000  # （即：尽可能多地提取关键点）。

        x, rh1, rw1 = self.preprocess_tensor(x)  # 对输入图像进行预处理，保证其尺寸能被 32 整除。

        M1, K1, H1 = self.net(x)  # 推理（Descriptors、Keypoint map logits、Reliability map）

        B, C, _H1, _W1 = M1.shape  # (B, 64, H/8, W/8)
        # 使用create_xy创建原始特征图上的坐标网格，然后乘以 8（因为特征图是原图的 1/8 大小），再扩展到 batch 维度。
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)

        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, H/8 * W/8, 64)
        H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)  # (B, H/8 * W/8)

        _, top_k = torch.topk(H1, k=min(len(H1[0]), top_k), dim=-1)  # 选择置信度最高的 top_k 个关键点的索引。

        # 从原始特征图上的坐标网格中提取对应的特征描述子。
        feats = torch.gather(M1, 1, top_k[..., None].expand(-1, -1, 64))  # (B, top_k, 64)
        # 从原始特征图上的坐标网格中提取对应的关键点坐标。
        mkpts = torch.gather(xy1, 1, top_k[..., None].expand(-1, -1, 2))  # (B, top_k, 2)
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)  # 乘以缩放因子，将关键点坐标转换回原始图像尺寸。

        return mkpts, feats

    # mkpts[0] = [
    # 	[100.5, 200.3], # 每个是一个[x, y]坐标
    # 	[150.2, 300.1],
    # 	...
    # 	[500.0, 600.0]
    # ]
    # feats[0] = [
    # 	[0.12, -0.34, ..., 0.78],  # 第一个关键点的 64 维描述子
    # 	[0.23, 0.45, ..., -0.67],  # 第二个关键点的 64 维描述子
    # 	...
    # ]

    #  对输入图像进行两个不同尺度（s1 和 s2）的缩放，分别提取稠密特征点与描述子，
    #  再将它们合并并返回统一结果，以增强模型对不同尺度目标的鲁棒性。
    def extract_dualscale(self, x, top_k, s1=0.6, s2=1.3):
        # 使用双线性插值方法将输入图像 x 分别缩放到 s1 和 s2 的尺寸
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

        B, _, _, _ = x.shape  # 获取输入图像的批次大小 B
        # 合理分配不同尺度图像的关键点提取数量，在保证尺度鲁棒性的同时，避免低分辨率图像引入过多噪声。
        mkpts_1, feats_1 = self.extractDense(x1, int(top_k * 0.20))  # 提取 x1 上的特征点和描述子
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k * 0.80))  # 提取 x2 上的特征点和描述子

        mkpts = torch.cat([mkpts_1 / s1, mkpts_2 / s2], dim=1)  # 将两个尺度的特征点合并，同时将它们的坐标缩放到原始图像尺寸
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1 / s1)  # 为每个特征点分配对应的尺度因子，也就是 1/s1
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1 / s2)  # 为每个特征点分配对应的尺度因子，也就是 1/s2
        sc = torch.cat([sc1, sc2], dim=1)  # 将两个尺度的特征点的尺度因子合并
        feats = torch.cat([feats_1, feats_2], dim=1)  # 将两个尺度的特征描述子合并

        return mkpts, sc, feats  # 返回合并后的特征点坐标、尺度因子和特征描述子

    # mkpts = torch.tensor([[
    # 	[100.5, 200.3],  # 来自尺度 s1=0.6 的关键点
    # 	[150.2, 300.1],  # 来自尺度 s1=0.6 的关键点
    #	......(200个特征点)
    # 	[200.7, 400.9],  # 来自尺度 s2=1.3 的关键点
    # 	[250.4, 500.6],  # 来自尺度 s2=1.3 的关键点
    # 	[300.8, 600.2]  # 来自尺度 s2=1.3 的关键点
    #   ......(800个特征点)
    # ]])

    # sc = torch.tensor([[1.6667, 1.6667, ..., 0.7692, 0.7692, 0.7692]])
    # 前200个特征点对应尺度因子 1/s1 = 1/0.6 ≈ 1.6667，800个特征点对应尺度因子 1/s2 = 1/1.3 ≈ 0.7692

    # feats = torch.tensor([[[0.12, -0.34, ..., 0.78],
    # [0.23, 0.45, ..., -0.67],
    # .....(200个特征描述子)
    # [-0.11, 0.56, ..., 0.44],
    # [0.33, -0.22, ..., 0.88],
    # [0.44, 0.11, ..., -0.99]]])
    # ......(800个特征描述子)

    #  用于统一图像输入格式，使其适配模型推理要求。
    def parse_input(self, x):
        if len(x.shape) == 3:  # 如果输入是三维张量（如 (H, W, C) 或 (C, H, W)），则增加一个 batch 维度
            x = x[None, ...]  # 转换为 (1, C, H, W)

        if isinstance(x, np.ndarray):  # 如果输入是 NumPy 数组
            x = torch.tensor(x).permute(0, 3, 1, 2) / 255
        # 把(B,H,W,C)转换为(B,C,H,W)，并将像素值归一化到[0,1]，并转换为 PyTorch 张量。

        return x  # 返回处理后的输入张量
