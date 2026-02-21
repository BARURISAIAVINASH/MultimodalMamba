# src/models/wavelet_layers.py - FIXED VERSION
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Function
import pywt

# 🔧 FIX: Mixed precision compatible wavelet functions
class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        # 🔧 FIX: Ensure matrices match input dtype
        matrix_Low_0 = matrix_Low_0.to(dtype=input.dtype, device=input.device)
        matrix_Low_1 = matrix_Low_1.to(dtype=input.dtype, device=input.device)
        matrix_High_0 = matrix_High_0.to(dtype=input.dtype, device=input.device)
        matrix_High_1 = matrix_High_1.to(dtype=input.dtype, device=input.device)
        
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        
        # 🔧 FIX: Ensure all gradients have same dtype
        grad_LL = grad_LL.to(dtype=matrix_Low_0.dtype)
        grad_LH = grad_LH.to(dtype=matrix_Low_0.dtype)
        grad_HL = grad_HL.to(dtype=matrix_Low_0.dtype)
        grad_HH = grad_HH.to(dtype=matrix_Low_0.dtype)
        
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        
        return grad_input, None, None, None, None

class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        # 🔧 FIX: Ensure matrices match input dtype
        matrix_Low_0 = matrix_Low_0.to(dtype=input_LL.dtype, device=input_LL.device)
        matrix_Low_1 = matrix_Low_1.to(dtype=input_LL.dtype, device=input_LL.device)
        matrix_High_0 = matrix_High_0.to(dtype=input_LL.dtype, device=input_LL.device)
        matrix_High_1 = matrix_High_1.to(dtype=input_LL.dtype, device=input_LL.device)
        
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        
        L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        
        # 🔧 FIX: Ensure gradient has same dtype
        grad_output = grad_output.to(dtype=matrix_Low_0.dtype)
        
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None

class DWT_2D(nn.Module):
    """Mixed precision compatible 2D Discrete Wavelet Transform"""
    def __init__(self, wavename='haar'):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        
        # 🔧 FIX: Initialize matrices as None, create on first forward
        self.matrix_low_0 = None
        self.matrix_low_1 = None
        self.matrix_high_0 = None
        self.matrix_high_1 = None

    def get_matrix(self, device, dtype):
        """Generate transformation matrices with correct device and dtype"""
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        
        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        
        # 🔧 FIX: Create tensors with correct dtype and device
        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=dtype, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=dtype, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=dtype, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=dtype, device=device)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        
        # 🔧 FIX: Always regenerate matrices with correct dtype/device
        self.get_matrix(input.device, input.dtype)
        
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1,
                                   self.matrix_high_0, self.matrix_high_1)

class IDWT_2D(nn.Module):
    """Mixed precision compatible 2D Inverse Discrete Wavelet Transform"""
    def __init__(self, wavename='haar'):
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        
        # 🔧 FIX: Initialize matrices as None
        self.matrix_low_0 = None
        self.matrix_low_1 = None
        self.matrix_high_0 = None
        self.matrix_high_1 = None

    def get_matrix(self, device, dtype):
        """Generate transformation matrices with correct device and dtype"""
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        
        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        
        # 🔧 FIX: Create tensors with correct dtype and device
        self.matrix_low_0 = torch.tensor(matrix_h_0, dtype=dtype, device=device)
        self.matrix_low_1 = torch.tensor(matrix_h_1, dtype=dtype, device=device)
        self.matrix_high_0 = torch.tensor(matrix_g_0, dtype=dtype, device=device)
        self.matrix_high_1 = torch.tensor(matrix_g_1, dtype=dtype, device=device)

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        
        # 🔧 FIX: Always regenerate matrices with correct dtype/device
        self.get_matrix(LL.device, LL.dtype)
        
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1,
                                   self.matrix_high_0, self.matrix_high_1)

# Keep all other wavelet classes unchanged
class Downsamplewave(nn.Module):
    """Downsampling with wavelet transform"""
    def __init__(self, wavename='haar'):
        super(Downsamplewave, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return torch.cat([LL, LH+HL+HH], dim=1)

class Downsamplewave1(nn.Module):
    """Downsampling with global pooling for attention"""
    def __init__(self, wavename='haar'):
        super(Downsamplewave1, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        LL = LL + LH + HL + HH
        result = torch.sum(LL, dim=[2, 3])  # Global sum pooling
        return result

class Waveletatt(nn.Module):
    """Channel-wise wavelet attention"""
    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution
        self.downsamplewavelet = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Downsamplewave1(wavename=wavename)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)
        y = xori * y.expand_as(xori)
        return y

class Waveletattspace(nn.Module):
    """Spatial wavelet attention"""
    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution
        self.downsamplewavelet = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Downsamplewave(wavename=wavename)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        xori = x
        B, C, H, W = x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        y = self.downsamplewavelet(x)
        y = self.fc(y)
        y = xori * y.expand_as(xori)
        return y
