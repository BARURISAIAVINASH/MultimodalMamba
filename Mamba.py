# ============================================================
# MultimodalMamba (DHCNN-FAM-ViM) - PyTorch reference code
# Implements:
#   - ViM Block: bidirectional ZOH-discretized SSM + Conv1D + gated fusion
#   - DHCNN: dilated hierarchical CNN (multi-scale conv features)
#   - FAM: frequency-aware attention (FFT-magnitude based, lightweight)
#   - DHCNN-FAM-ViM block with 2D<->1D token interface (patchify)
#   - Stacked stages with channel doubling + GAP + Softmax classifier
#
# NOTE:
# - This is a clean, runnable "paper-aligned" implementation without relying
#   on external mamba libraries. The SSM here uses a diagonal state matrix
#   (efficient + stable) and a bidirectional scan (forward + backward).
# - You can swap SSMDiagonal with a faster CUDA selective-scan later.
# ============================================================

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


class RMSNorm(nn.Module):
    """Simple RMSNorm over last dimension."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


# -----------------------------
# 1D Depthwise Separable Conv
# -----------------------------
class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, groups=dim, bias=bias)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        return self.pw(self.dw(x))


# -----------------------------
# Diagonal ZOH-Discretized SSM (efficient reference)
# -----------------------------
class SSMDiagonal(nn.Module):
    """
    Diagonal continuous-time SSM:
        dh/dt = A h + B x
        y     = C h
    ZOH discretization with step dt.

    Shapes:
      x: (B, L, D_in)  -> internally projects to state N
      output y: (B, L, D_out)  (here D_out == D_model typically)
    """
    def __init__(
        self,
        d_model: int,
        n_state: int = 64,
        dt_init: float = 1.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state
        self.bidirectional = bidirectional

        # Input/output projections (to/from state space)
        self.in_proj = nn.Linear(d_model, n_state, bias=False)
        self.out_proj = nn.Linear(n_state, d_model, bias=False)

        # Diagonal continuous-time A parameterized in log-space for stability
        # We keep A negative: A = -exp(log_A)
        self.log_A = nn.Parameter(torch.randn(n_state) * 0.02)

        # B and C (diagonal-ish but learnable vectors) in state space
        self.B = nn.Parameter(torch.randn(n_state) * 0.02)
        self.C = nn.Parameter(torch.randn(n_state) * 0.02)

        # Step size dt (positive)
        self.log_dt = nn.Parameter(torch.tensor(math.log(dt_init), dtype=torch.float32))

    def _zoh_discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For diagonal A:
          Abar = exp(A*dt)
          Bbar = integral_0^dt exp(A*t) dt * B = (A^-1 (Abar - I)) * B
        """
        dt = torch.exp(self.log_dt)  # scalar
        A = -torch.exp(self.log_A)   # (N,) negative
        A_dt = A * dt
        Abar = torch.exp(A_dt)       # (N,)

        # Safe compute (Abar - 1)/A for A near 0
        # Use series approx when |A_dt| small: (exp(A_dt)-1)/A ≈ dt
        eps = 1e-6
        denom = torch.where(A.abs() < eps, torch.ones_like(A), A)
        frac = (Abar - 1.0) / denom
        frac = torch.where(A.abs() < eps, dt.expand_as(frac), frac)

        Bbar = frac * self.B  # (N,)
        return Abar, Bbar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns y: (B, L, D)
        """
        B, L, D = x.shape
        assert D == self.d_model, f"Expected D={self.d_model}, got {D}"

        Abar, Bbar = self._zoh_discretize()  # (N,), (N,)
        Abar = Abar.to(x.dtype).to(x.device)
        Bbar = Bbar.to(x.dtype).to(x.device)
        C = self.C.to(x.dtype).to(x.device)

        u = self.in_proj(x)  # (B, L, N)

        # Scan recurrence: h_t = Abar*h_{t-1} + Bbar*u_t
        h = torch.zeros(B, self.n_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = h * Abar + u[:, t, :] * Bbar
            y_t = h * C
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, N)
        y = self.out_proj(y)        # (B, L, D)
        return y


# -----------------------------
# ViM Block (bidirectional SSM + Conv1D + gating)
# -----------------------------
class ViMBlock(nn.Module):
    """
    Matches your equations (24)-(27) at a module level:
      x,z = Linear(Norm(T))
      x~  = Conv1D(x)
      y_f = SSM->(x~), y_b = SSM<-(x~)
      T'  = Linear((y_f + y_b) ⊙ sigmoid(z))
    Includes residual connection.
    """
    def __init__(
        self,
        d_model: int,
        n_state: int = 64,
        conv_kernel: int = 3,
        dropout: float = 0.0,
        norm: str = "rms",
    ):
        super().__init__()
        self.norm = RMSNorm(d_model) if norm == "rms" else nn.LayerNorm(d_model)
        self.in_linear = nn.Linear(d_model, 2 * d_model, bias=True)

        # Conv1D expects (B, D, L)
        self.conv = DepthwiseSeparableConv1D(d_model, kernel_size=conv_kernel)

        self.ssm_fwd = SSMDiagonal(d_model=d_model, n_state=n_state, bidirectional=False)
        self.ssm_bwd = SSMDiagonal(d_model=d_model, n_state=n_state, bidirectional=False)

        self.out_linear = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        # T: (B, L, D)
        residual = T
        Tn = self.norm(T)
        xz = self.in_linear(Tn)                 # (B, L, 2D)
        x, z = torch.chunk(xz, 2, dim=-1)       # (B, L, D), (B, L, D)
        gate = torch.sigmoid(z)

        # local conv over sequence
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, D)

        # forward SSM
        y_f = self.ssm_fwd(x_conv)              # (B, L, D)

        # backward SSM: reverse time, run SSM, reverse back
        x_rev = torch.flip(x_conv, dims=[1])
        y_b = self.ssm_bwd(x_rev)
        y_b = torch.flip(y_b, dims=[1])

        y = (y_f + y_b) * gate                  # gated fusion
        y = self.out_linear(y)
        y = self.drop(y)
        return residual + y


# -----------------------------
# DHCNN (Dilated Hierarchical CNN) - lightweight reference
# -----------------------------
class DHCNN(nn.Module):
    """
    A simple dilated hierarchical CNN block:
      - pointwise expand
      - stack dilated depthwise convs (multi-scale)
      - fuse + project back
    """
    def __init__(self, in_ch: int, out_ch: int, dilations: Tuple[int, ...] = (1, 2, 3), dropout: float = 0.0):
        super().__init__()
        mid = out_ch
        self.pw_in = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(mid)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid, mid, kernel_size=3, padding=d, dilation=d, groups=mid, bias=False),
                nn.BatchNorm2d(mid),
                nn.GELU(),
            )
            for d in dilations
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(mid * len(dilations), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.drop = nn.Dropout2d(dropout)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        res = self.skip(x)
        x = self.bn_in(self.pw_in(x))
        x = F.gelu(x)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.drop(x)
        return res + x


# -----------------------------
# FAM (Frequency Attention Module) - FFT-magnitude based
# -----------------------------
class FAM(nn.Module):
    """
    Frequency attention:
      - compute FFT magnitude map (rfft2) -> pooled descriptor
      - produce channel weights
      - reweight spatial features
    """
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(ch, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, ch, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # FFT magnitude
        Xf = torch.fft.rfft2(x, norm="ortho")            # (B, C, H, W//2+1) complex
        mag = torch.abs(Xf)                              # (B, C, H, Wf)

        # Global frequency descriptor (channel-wise)
        desc = mag.mean(dim=(2, 3))                      # (B, C)
        w = torch.sigmoid(self.mlp(desc)).view(B, C, 1, 1)
        return x * w


# -----------------------------
# Patchify / Unpatchify (2D <-> token sequence)
# -----------------------------
def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    x: (B, C, H, W) with H,W divisible by p
    returns tokens: (B, L, p*p*C), L=(H/p)*(W/p)
    """
    B, C, H, W = x.shape
    assert H % p == 0 and W % p == 0, f"H,W must be divisible by p={p}, got {(H,W)}"
    hp, wp = H // p, W // p
    x = x.view(B, C, hp, p, wp, p)           # (B,C,hp,p,wp,p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B,hp,wp,p,p,C)
    tokens = x.view(B, hp * wp, p * p * C)
    return tokens


def unpatchify(tokens: torch.Tensor, p: int, h: int, w: int, c: int) -> torch.Tensor:
    """
    tokens: (B, L, p*p*C), L=(H/p)*(W/p)
    returns x: (B, C, H, W)
    """
    B, L, D = tokens.shape
    assert D == p * p * c, f"Token dim mismatch: got {D}, expected {p*p*c}"
    hp, wp = h // p, w // p
    assert L == hp * wp, f"Token length mismatch: got {L}, expected {hp*wp}"

    x = tokens.view(B, hp, wp, p, p, c)      # (B,hp,wp,p,p,C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B,C,hp,p,wp,p)
    x = x.view(B, c, h, w)
    return x


# -----------------------------
# DHCNN-FAM-ViM (Mamba interface) block
# -----------------------------
class DHCNN_FAM_ViM_Block(nn.Module):
    """
    Pipeline:
      x -> DHCNN -> FAM -> patchify -> Conv1D proj -> ViM -> inv Conv1D -> unpatchify
    """
    def __init__(
        self,
        ch: int,
        p: int = 7,
        d_model: int = 256,
        n_state: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.ch = ch
        self.d_model = d_model

        self.dhcnn = DHCNN(in_ch=ch, out_ch=ch, dilations=(1, 2, 3), dropout=dropout)
        self.fam = FAM(ch)

        token_dim = p * p * ch

        # sequence projection convs (kernel=3)
        self.seq_in = nn.Conv1d(token_dim, d_model, kernel_size=3, padding=1, bias=True)
        self.vim = ViMBlock(d_model=d_model, n_state=n_state, conv_kernel=3, dropout=dropout)
        self.seq_out = nn.Conv1d(d_model, token_dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.ch

        x = self.dhcnn(x)
        x = self.fam(x)

        # 2D -> tokens
        tokens = patchify(x, self.p)  # (B, L, token_dim)
        L = tokens.shape[1]
        token_dim = tokens.shape[2]

        # tokens -> model sequence (Conv1D expects (B, token_dim, L))
        seq = tokens.transpose(1, 2)            # (B, token_dim, L)
        seq = self.seq_in(seq)                  # (B, d_model, L)
        seq = seq.transpose(1, 2)               # (B, L, d_model)

        # ViM (bidirectional SSM)
        seq = self.vim(seq)                     # (B, L, d_model)

        # back to tokens
        seq = seq.transpose(1, 2)               # (B, d_model, L)
        tokens = self.seq_out(seq).transpose(1, 2)  # (B, L, token_dim)

        # tokens -> 2D
        x = unpatchify(tokens, self.p, H, W, C) # (B, C, H, W)
        return x


# -----------------------------
# Full MultimodalMamba backbone + classifier
# -----------------------------
class MultimodalMamba(nn.Module):
    """
    Stages S1..Sk:
      - optional stem to desired channels
      - per stage: N blocks, channels double each stage
      - final: GAP -> FC -> logits
    """
    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 6,
        stages: List[int] = [1, 1, 2, 2],   # blocks per stage
        base_ch: int = 64,
        p: int = 7,
        d_model: int = 256,
        n_state: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.p = p

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )

        ch = base_ch
        stage_modules = []
        for si, n_blocks in enumerate(stages):
            blocks = []
            for _ in range(n_blocks):
                blocks.append(DHCNN_FAM_ViM_Block(
                    ch=ch,
                    p=p,
                    d_model=d_model,
                    n_state=n_state,
                    dropout=dropout
                ))
            stage_modules.append(nn.Sequential(*blocks))

            # Downsample + channel double between stages (except last)
            if si != len(stages) - 1:
                stage_modules.append(nn.Sequential(
                    nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch * 2),
                    nn.GELU(),
                ))
                ch *= 2

        self.backbone = nn.Sequential(*stage_modules)
        self.head_norm = nn.BatchNorm2d(ch)
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W) ; ensure H,W divisible by p at all ViM blocks
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head_norm(x)

        # GAP + FC
        x = x.mean(dim=(2, 3))     # (B, C)
        logits = self.classifier(x)
        return logits


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example: spectrogram/CSI images (B,C,H,W)
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)

    model = MultimodalMamba(
        in_ch=C,
        num_classes=10,
        stages=[1, 1, 1],     # keep small for test
        base_ch=64,
        p=7,
        d_model=256,
        n_state=64,
        dropout=0.1,
    )

    with torch.no_grad():
        y = model(x)
    print("logits:", y.shape)  # (B, num_classes)