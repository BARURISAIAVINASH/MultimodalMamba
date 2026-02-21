# ============================================================
# Dilated Hierarchical CNN (DHCNN) Block (PyTorch)
# As described:
# - Input x: (B, C, H, W)
# - Split channels into s splits (pad channels if C not divisible by s)
# - Hierarchical processing across splits:
#   - Fixed 3x3 depthwise conv with increasing dilation d = 1,2,3,... (per level)
#   - then pointwise conv (1x1), BN, ReLU
#   - Split output (w channels) into two halves:
#       * one half goes directly to output
#       * other half is carried and concatenated with next split φ_{l+1}
# - Repeat across l = 1..s
# - Optional skip connection for stability
# - Return concatenated outputs
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Depthwise + Pointwise with dilation
# -----------------------------
class DilatedDepthwiseSeparableConv2d(nn.Module):
    """
    Fixed 3x3 depthwise conv with dilation, followed by 1x1 pointwise conv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        k = 3
        pad = dilation  # for 3x3 with dilation, "same" padding = dilation

        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=k,
            stride=1,
            padding=pad,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# -----------------------------
# One DHCNN hierarchical stage
# -----------------------------
class DHCNNStage(nn.Module):
    def __init__(self, in_ch: int, w: int, dilation: int):
        super().__init__()
        self.conv = DilatedDepthwiseSeparableConv2d(in_ch, w, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(w)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@dataclass
class DHCNNConfig:
    s: int = 4                  # number of splits/levels
    K: int = 2                  # input-to-output ratio (bigger => smaller w)
    dilations: Optional[List[int]] = None  # e.g., [1,2,3,4] or [1,2,3,1,...]
    keep_last_carry: bool = True            # include final carry in output
    use_global_skip: bool = True            # x + proj(out)
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1


class DHCNNBlock(nn.Module):
    """
    Implements the Dilated Hierarchical CNN block.

    x: (B,C,H,W)
    - pad channels so divisible by s
    - split into s chunks: φ1..φs
    - for l=1..s:
        inp = φ1                 (l=1)
        inp = carry ⊕ φ_l        (l>1)
        y = stage_l(inp) -> (B,w,H,W)
        direct, carry = split(y)  (half-half)
        collect direct
      optionally collect last carry too
    - concat collected parts -> y_out
    - optional skip: y_out + proj(x_padded)
    """
    def __init__(self, in_channels: int, cfg: DHCNNConfig):
        super().__init__()
        assert cfg.s >= 1
        assert cfg.K >= 1

        self.in_channels = in_channels
        self.cfg = cfg

        # pad to be divisible by s
        self.pad_channels = (cfg.s - (in_channels % cfg.s)) % cfg.s
        self.c_prime = in_channels + self.pad_channels
        self.chunk_ch = self.c_prime // cfg.s

        # width per stage (w)
        self.w = max(2, self.chunk_ch // cfg.K)  # ensure at least 2 so split works

        # dilation schedule
        if cfg.dilations is None:
            # paper mentions d=1,2,3 progressively; for more splits continue 1,2,3,4...
            dilations = list(range(1, cfg.s + 1))
        else:
            assert len(cfg.dilations) == cfg.s
            dilations = cfg.dilations
        self.dilations = dilations

        # carry channels (half of w)
        self.carry_ch = self.w // 2
        self.direct_ch = self.w - self.carry_ch

        # build stages
        self.stages = nn.ModuleList()
        for l in range(cfg.s):
            in_ch = self.chunk_ch if l == 0 else (self.chunk_ch + self.carry_ch)
            st = DHCNNStage(in_ch=in_ch, w=self.w, dilation=dilations[l])
            # set BN params if needed
            st.bn.eps = cfg.bn_eps
            st.bn.momentum = cfg.bn_momentum
            self.stages.append(st)

        # output channels after concatenation
        out_ch = cfg.s * self.direct_ch + (self.carry_ch if cfg.keep_last_carry else 0)
        self.out_channels = out_ch

        # skip projection to match y_out channels (using padded input)
        self.skip_proj = None
        if cfg.use_global_skip:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(self.c_prime, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum),
            )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_channels == 0:
            return x
        b, c, h, w = x.shape
        pad = x.new_zeros((b, self.pad_channels, h, w))
        return torch.cat([x, pad], dim=1)

    def _split_direct_carry(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split (B,w,H,W) into direct and carry parts along channel.
        """
        direct = y[:, :self.direct_ch, :, :]
        carry = y[:, self.direct_ch:self.direct_ch + self.carry_ch, :, :]
        return direct, carry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1] == self.in_channels

        x_pad = self._pad(x)                 # (B,C',H,W)
        chunks = list(torch.chunk(x_pad, self.cfg.s, dim=1))  # φ1..φs, each (B, C'/s, H, W)

        outs: List[torch.Tensor] = []
        carry: Optional[torch.Tensor] = None

        for l in range(self.cfg.s):
            phi = chunks[l]
            inp = phi if carry is None else torch.cat([carry, phi], dim=1)  # carry ⊕ φ_l
            y = self.stages[l](inp)  # (B,w,H,W)
            direct, carry = self._split_direct_carry(y)
            outs.append(direct)

        if self.cfg.keep_last_carry and carry is not None and carry.shape[1] > 0:
            outs.append(carry)

        y_out = torch.cat(outs, dim=1)  # (B,out_channels,H,W)

        if self.skip_proj is not None:
            y_out = F.relu(y_out + self.skip_proj(x_pad), inplace=True)

        return y_out


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 2, 30, 64, 64
    cfg = DHCNNConfig(s=4, K=2, dilations=[1, 2, 3, 4], keep_last_carry=True, use_global_skip=True)

    x = torch.randn(B, C, H, W)
    block = DHCNNBlock(in_channels=C, cfg=cfg)

    y = block(x)
    print("x:", x.shape)
    print("y:", y.shape)
    print("pad_channels:", block.pad_channels, "C':", block.c_prime)
    print("chunk_ch:", block.chunk_ch, "w:", block.w, "out_channels:", block.out_channels)