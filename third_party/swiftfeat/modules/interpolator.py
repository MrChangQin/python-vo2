"""
此模块用于实现插值算法，提供给XFeat使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolateSparse2d(nn.Module):
    """
    在给定的稀疏2D位置高效插值张量
    用于在特征图上对任意位置进行插值
    """
    def __init__(self, mode = 'bicubic', align_corners = False):
        """
        初始化插值器
        参数:
            mode: 插值模式，默认为'bicubic'（双三次插值）
            align_corners: 是否对齐角点，默认为False
        """
        super().__init__()
        self.mode = mode  # 插值模式
        self.align_corners = align_corners  # 角点对齐设置

    def normgrid(self, x, H, W):
        """
        将坐标从像素坐标归一化到[-1, 1]范围
        参数:
            x: 输入坐标，形状为[B, N, 2]
            H, W: 原始图像的高度和宽度
        返回:
        # 调整输出维度并返回
        # 使用grid_sample进行插值
        # 将坐标归一化并调整维度
        前向传播，在给定位置进行插值
        参数:
            x: 输入特征图，形状为[B, C, H, W]
            pos: 要插值的位置，形状为[B, N, 2]
            H, W: 原始图像的高度和宽度
        返回:
            插值后的特征，形状为[B, N, C]
            归一化后的坐标，形状为[B, N, 2]
        """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.
        # device = x.device 和 dtype = x.dtype:
        # 确保生成的张量与输入x在相同的设备和数据类型上，避免不必要的设备间数据传输和类型转换
        # 然后通过 2. * x - 1. 的变换，将 [0, 1] 的范围映射到 [-1, 1] 的范围，这是 grid_sample 函数所期望的输入范围。

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)