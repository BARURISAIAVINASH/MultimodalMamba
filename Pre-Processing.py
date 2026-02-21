"""
============================================================
Multimodal Preprocessing + Feature Extraction + Fusion (PyTorch)
- Window length: 1.04s
- 50% overlap sliding window (train + inference)
- RGB: 16 uniformly sampled frames (appearance) + dense optical flow (motion)
- Depth: background suppression + contrast mapping + SFI (sum over 16 frames) -> 64x64
- IMU: 50 Hz -> 52 samples per window, 7 channels -> 4 stats/channel -> 28 features/time
       stack over time => 28x52 "signal image" -> resize 64x64
- Feature extraction: depthwise separable CNN streams
- Fusion: concat RGB(app)+RGB(flow)+Depth+IMU -> MultimodalMamba-like fusion -> classifier

NOTE:
- This is a complete, detailed, engineering-ready reference implementation.
- You can plug in your own data reading code (videos, depth frames, imu arrays).
============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# OpenCV is used for optical flow; install: pip install opencv-python
try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
# 0) Windowing utilities
# ============================================================
@dataclass
class WindowConfig:
    window_seconds: float = 1.04
    overlap: float = 0.5

    rgb_fps: int = 30         # adjust to your RGB fps
    depth_fps: int = 30       # adjust to your depth fps
    imu_hz: int = 50          # given in paper

    rgb_frames_T: int = 16    # sample 16 frames uniformly
    depth_frames_T: int = 16  # accumulate 16 frames for SFI
    out_size: int = 64        # resize to 64x64

    # IMU: 1.04s * 50Hz ≈ 52 samples
    imu_steps: int = 52


def sliding_window_indices(num_steps: int, win_steps: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) windows in step-units (end exclusive).
    """
    assert 0.0 <= overlap < 1.0
    hop = max(1, int(round(win_steps * (1.0 - overlap))))
    windows = []
    s = 0
    while s + win_steps <= num_steps:
        windows.append((s, s + win_steps))
        s += hop
    return windows


def uniform_sample_indices(start: int, end: int, num: int) -> np.ndarray:
    """
    Uniformly sample 'num' indices from [start, end) inclusive-ish.
    """
    if num <= 1:
        return np.array([start], dtype=np.int64)
    # sample in continuous space then round
    xs = np.linspace(start, end - 1, num=num)
    return np.clip(np.round(xs).astype(np.int64), start, end - 1)


# ============================================================
# 1) RGB preprocessing
#    - appearance: 16 sampled RGB frames
#    - flow: dense optical flow across consecutive frames in window
# ============================================================
def rgb_to_tensor_uint8(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    frame_bgr: HxWx3 uint8 (OpenCV default BGR)
    returns: 3xHxW float in [0,1]
    """
    frame_rgb = frame_bgr[..., ::-1].copy()
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return t


def resize_tensor_image(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    x: CxHxW or BxCxHxW
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
        y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return y.squeeze(0)
    elif x.dim() == 4:
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    else:
        raise ValueError("Expected 3D or 4D tensor image")


def compute_dense_optical_flow_farneback(frames_bgr: List[np.ndarray]) -> np.ndarray:
    """
    frames_bgr: list length T, each HxWx3 uint8
    returns: flow_stack as float32 array of shape (T-1, H, W, 2)
             (u,v) per pixel between consecutive frames.
    """
    if cv2 is None:
        raise RuntimeError("cv2 not available. Install: pip install opencv-python")

    flows = []
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames_bgr)):
        gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow.astype(np.float32))
        prev_gray = gray
    return np.stack(flows, axis=0)  # (T-1, H, W, 2)


def flow_to_image_tensor(flow_hw2: np.ndarray) -> torch.Tensor:
    """
    flow_hw2: HxWx2 float32
    Convert flow to 2-channel tensor (u,v) normalized roughly.
    You can change normalization to match your pipeline.
    returns: 2xHxW float
    """
    flow = torch.from_numpy(flow_hw2).permute(2, 0, 1).float()  # 2xHxW
    # simple robust normalization
    # scale by 95th percentile magnitude to keep values stable
    mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
    s = torch.quantile(mag.flatten(), 0.95).clamp(min=1e-6)
    flow = flow / s
    flow = flow.clamp(-3.0, 3.0)
    return flow


# ============================================================
# 2) Depth preprocessing
#    - background suppression (ROI mask)
#    - piecewise linear contrast mapping f(.)
#    - SFI: sum of 16 mapped depth frames -> resize 64x64
# ============================================================
@dataclass
class DepthConfig:
    # For ROI: you can pass a binary mask or provide bounding box.
    # If no ROI mask, we treat all pixels as ROI.
    use_roi_mask: bool = False

    # Contrast mapping: piecewise linear / clamp mapping
    # Example: clamp depth to [d_min, d_max] then map to [0,1]
    d_min: float = 0.2
    d_max: float = 4.0

    # Optional gamma to enhance contrast after mapping
    gamma: float = 1.0


def depth_background_suppression(depth: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    depth: HxW float32
    roi_mask: HxW bool/0-1 mask, True means ROI
    """
    if roi_mask is None:
        return depth
    out = depth.copy()
    out[~roi_mask.astype(bool)] = 0.0
    return out


def depth_contrast_mapping(depth: np.ndarray, cfg: DepthConfig) -> np.ndarray:
    """
    Piecewise linear mapping f(D): clamp depth to [d_min, d_max], then normalize to [0,1].
    If depth is 0 (background), stays 0.
    """
    d = depth.copy()
    bg = (d <= 0.0)
    d = np.clip(d, cfg.d_min, cfg.d_max)
    d = (d - cfg.d_min) / max(cfg.d_max - cfg.d_min, 1e-6)
    if cfg.gamma != 1.0:
        d = np.power(d, cfg.gamma)
    d[bg] = 0.0
    return d.astype(np.float32)


def make_sfi(depth_frames: List[np.ndarray], roi_mask: Optional[np.ndarray], cfg: DepthConfig) -> torch.Tensor:
    """
    depth_frames: list length T(=16), each HxW float32
    returns: 1xHxW tensor float in [0, T] then normalized to [0,1]
    """
    acc = None
    for d in depth_frames:
        d1 = depth_background_suppression(d, roi_mask)
        it = depth_contrast_mapping(d1, cfg)
        acc = it if acc is None else (acc + it)

    # Normalize SFI to [0,1] by dividing by T (as a simple option)
    T = len(depth_frames)
    sfi = (acc / max(T, 1)).astype(np.float32)
    t = torch.from_numpy(sfi).unsqueeze(0)  # 1xHxW
    return t


# ============================================================
# 3) IMU preprocessing: signal image
#    - 7 channels: ax,ay,az,gx,gy,gz,smag
#    - For each time step compute 4 stats per channel over a short sub-window
#      -> 28 features per time step
#    - Stack over 52 steps => 28x52 matrix -> resize 64x64
# ============================================================
def imu_compute_smag(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    return np.sqrt(ax**2 + ay**2 + az**2)


def stats_4(x: np.ndarray) -> np.ndarray:
    """
    4 simple stats from a 1D array:
    mean, std, min, max
    """
    return np.array([x.mean(), x.std(ddof=0), x.min(), x.max()], dtype=np.float32)


def imu_make_feature_matrix(
    imu_52x6: np.ndarray,
    local_stat_window: int = 5,
) -> np.ndarray:
    """
    imu_52x6: (52,6) channels = [ax,ay,az,gx,gy,gz]
    Build per-time-step 28 features:
      7 channels * 4 stats, where stats are computed over a small local window around t.
    returns: (28,52) float32
    """
    assert imu_52x6.shape[0] == 52 and imu_52x6.shape[1] == 6, "Expect (52,6)"

    ax, ay, az = imu_52x6[:, 0], imu_52x6[:, 1], imu_52x6[:, 2]
    gx, gy, gz = imu_52x6[:, 3], imu_52x6[:, 4], imu_52x6[:, 5]
    smag = imu_compute_smag(ax, ay, az)

    chans = [ax, ay, az, gx, gy, gz, smag]
    T = 52
    half = local_stat_window // 2

    feats = np.zeros((28, T), dtype=np.float32)

    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        vecs = []
        for ch in chans:
            vecs.append(stats_4(ch[s:e]))
        vec = np.concatenate(vecs, axis=0)  # (28,)
        feats[:, t] = vec

    # robust normalize each feature row (optional but helps)
    med = np.median(feats, axis=1, keepdims=True)
    mad = np.median(np.abs(feats - med), axis=1, keepdims=True) + 1e-6
    feats = (feats - med) / (1.4826 * mad)
    return feats.astype(np.float32)


def imu_feature_matrix_to_image(feat_28x52: np.ndarray) -> torch.Tensor:
    """
    feat_28x52 -> 1x28x52 tensor
    """
    t = torch.from_numpy(feat_28x52).unsqueeze(0)  # 1x28x52
    return t


def resize_signal_image(x_1xhxw: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    """
    Resize 1xHxW to 1xout_sizexout_size using bilinear (treat like image).
    """
    return resize_tensor_image(x_1xhxw, out_size)


# ============================================================
# 4) Depthwise separable CNN backbone (small)
# ============================================================
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1):
        super().__init__()
        pad = (k - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=stride, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class SmallDSCNN(nn.Module):
    """
    A lightweight DS-CNN feature extractor that outputs a feature vector.
    """
    def __init__(self, in_ch: int, base: int = 32, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSeparableConv(base, base, 3, stride=1),
            DWSeparableConv(base, base * 2, 5, stride=2),
            DWSeparableConv(base * 2, base * 2, 3, stride=1),
            DWSeparableConv(base * 2, base * 4, 7, stride=2),
            DWSeparableConv(base * 4, base * 4, 3, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)  # (B, out_dim)


# ============================================================
# 5) Multimodal "Mamba-like" fusion network (MMN)
#    - Takes modality vectors, builds a short token sequence, mixes with
#      gated depthwise 1D conv + residual (a practical Mamba-style mixer)
# ============================================================
class GatedConvMixer1D(nn.Module):
    """
    Simple sequence mixer:
      y = x + Conv1D(GELU(W1(x)) * sigmoid(W2(x)))
    where Conv1D is depthwise (groups=dim) for efficiency.
    """
    def __init__(self, dim: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad, groups=dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        h = self.norm(x)
        a = F.gelu(self.w1(h))
        g = torch.sigmoid(self.w2(h))
        z = a * g  # (B,T,D)

        # depthwise conv expects (B,D,T)
        z = z.transpose(1, 2)
        z = self.dwconv(z)
        z = z.transpose(1, 2)

        z = self.drop(z)
        return x + z


class MultimodalMambaNetwork(nn.Module):
    """
    Input: modality vectors [rgb_app, rgb_flow, depth, imu] each (B, D)
    Build tokens (T=4) => mix => pool => classify
    """
    def __init__(self, dim: int, num_layers: int = 4, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.mixers = nn.ModuleList([GatedConvMixer1D(dim, kernel=3, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T=4, D)
        """
        x = tokens
        for m in self.mixers:
            x = m(x)
        x = self.norm(x)

        # global token pooling (mean)
        z = x.mean(dim=1)  # (B,D)
        return self.cls(z)


# ============================================================
# 6) Full model: RGB(appearance+flow), Depth(SFI), IMU(signal image) + Fusion
# ============================================================
class MultimodalHARModel(nn.Module):
    def __init__(self, feat_dim: int = 256, num_classes: int = 10):
        super().__init__()

        # RGB appearance stream: input 3x64x64
        self.rgb_app_net = SmallDSCNN(in_ch=3, base=32, out_dim=feat_dim)

        # Optical flow stream: input 2x64x64 (u,v)
        self.rgb_flow_net = SmallDSCNN(in_ch=2, base=32, out_dim=feat_dim)

        # Depth SFI stream: input 1x64x64
        self.depth_net = SmallDSCNN(in_ch=1, base=32, out_dim=feat_dim)

        # IMU signal image stream: input 1x64x64
        self.imu_net = SmallDSCNN(in_ch=1, base=32, out_dim=feat_dim)

        # Fusion MMN
        self.mmn = MultimodalMambaNetwork(dim=feat_dim, num_layers=4, num_classes=num_classes, dropout=0.1)

    def forward(self, rgb_app: torch.Tensor, rgb_flow: torch.Tensor, depth_sfi: torch.Tensor, imu_img: torch.Tensor):
        """
        rgb_app: (B,3,64,64)
        rgb_flow: (B,2,64,64)
        depth_sfi: (B,1,64,64)
        imu_img: (B,1,64,64)
        """
        f_app = self.rgb_app_net(rgb_app)
        f_flow = self.rgb_flow_net(rgb_flow)
        f_depth = self.depth_net(depth_sfi)
        f_imu = self.imu_net(imu_img)

        tokens = torch.stack([f_app, f_flow, f_depth, f_imu], dim=1)  # (B,4,D)
        logits = self.mmn(tokens)
        return logits


# ============================================================
# 7) Dataset skeleton
#    You plug your own reader: RGB frames, depth frames, IMU array.
# ============================================================
class MultimodalWindowDataset(Dataset):
    """
    Expects each sample item provides:
      - rgb_frames: list/array of BGR frames (N_rgb, H, W, 3) uint8
      - depth_frames: list/array (N_depth, H, W) float32
      - imu_stream: (N_imu, 6) float32 for [ax,ay,az,gx,gy,gz]
      - label: int

    It will create sliding windows aligned by time, then output:
      rgb_app_img: (3,64,64) from 16 sampled frames (we take the middle sampled frame for appearance)
      rgb_flow_img: (2,64,64) from dense flow aggregated over window
      depth_sfi_img: (1,64,64)
      imu_img: (1,64,64)
    """
    def __init__(
        self,
        items: List[Dict],
        cfg: WindowConfig,
        depth_cfg: DepthConfig,
        roi_masks: Optional[Dict[int, np.ndarray]] = None,  # optional per-item ROI mask
    ):
        super().__init__()
        self.items = items
        self.cfg = cfg
        self.depth_cfg = depth_cfg
        self.roi_masks = roi_masks or {}

        # Build index of (item_idx, rgb_start, rgb_end, depth_start, depth_end, imu_start, imu_end)
        self.index: List[Tuple[int, int, int, int, int, int, int]] = []
        for i, it in enumerate(items):
            n_rgb = len(it["rgb_frames"])
            n_depth = len(it["depth_frames"])
            n_imu = it["imu_stream"].shape[0]

            # compute window lengths in steps
            win_rgb = int(round(cfg.window_seconds * cfg.rgb_fps))
            win_depth = int(round(cfg.window_seconds * cfg.depth_fps))
            win_imu = cfg.imu_steps  # fixed 52

            rgb_wins = sliding_window_indices(n_rgb, win_rgb, cfg.overlap)
            depth_wins = sliding_window_indices(n_depth, win_depth, cfg.overlap)
            imu_wins = sliding_window_indices(n_imu, win_imu, cfg.overlap)

            # Align by window count (simple approach):
            m = min(len(rgb_wins), len(depth_wins), len(imu_wins))
            for w in range(m):
                rs, re = rgb_wins[w]
                ds, de = depth_wins[w]
                is_, ie = imu_wins[w]
                self.index.append((i, rs, re, ds, de, is_, ie))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        item_i, rs, re, ds, de, is_, ie = self.index[idx]
        it = self.items[item_i]
        label = int(it["label"])

        rgb_frames_win = it["rgb_frames"][rs:re]    # list of BGR uint8 frames
        depth_frames_win = it["depth_frames"][ds:de]  # list of HxW float32
        imu_win = it["imu_stream"][is_:ie]          # (52,6)

        # ---- RGB appearance (16 sampled frames, choose one representative or stack)
        sel = uniform_sample_indices(0, len(rgb_frames_win), self.cfg.rgb_frames_T)
        sampled = [rgb_frames_win[int(j)] for j in sel]

        # Option: use the middle sampled frame as "appearance frame"
        app_frame = sampled[len(sampled) // 2]
        rgb_app = rgb_to_tensor_uint8(app_frame)
        rgb_app = resize_tensor_image(rgb_app, self.cfg.out_size)  # (3,64,64)

        # ---- RGB optical flow: compute dense flow over entire window,
        # then aggregate flow into a single 2-channel "flow image"
        # A simple approach: average flow over time
        flow_img = self._make_flow_image(rgb_frames_win)

        # ---- Depth SFI
        roi = self.roi_masks.get(item_i, None)
        depth_sel = uniform_sample_indices(0, len(depth_frames_win), self.cfg.depth_frames_T)
        depth_16 = [depth_frames_win[int(j)].astype(np.float32) for j in depth_sel]
        sfi = make_sfi(depth_16, roi, self.depth_cfg)  # 1xHxW
        sfi = resize_tensor_image(sfi, self.cfg.out_size)  # 1x64x64

        # ---- IMU signal image
        imu_52x6 = imu_win.astype(np.float32)
        feat_28x52 = imu_make_feature_matrix(imu_52x6, local_stat_window=5)
        imu_img = imu_feature_matrix_to_image(feat_28x52)  # 1x28x52
        imu_img = resize_signal_image(imu_img, out_size=self.cfg.out_size)  # 1x64x64

        return {
            "rgb_app": rgb_app,
            "rgb_flow": flow_img,
            "depth_sfi": sfi,
            "imu_img": imu_img,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _make_flow_image(self, rgb_frames_win: List[np.ndarray]) -> torch.Tensor:
        if len(rgb_frames_win) < 2:
            # fallback: zero flow
            h, w = rgb_frames_win[0].shape[:2]
            flow = torch.zeros(2, h, w, dtype=torch.float32)
            return resize_tensor_image(flow, self.cfg.out_size)

        flow_stack = compute_dense_optical_flow_farneback(rgb_frames_win)  # (T-1,H,W,2)
        # average across time -> HxWx2
        flow_mean = flow_stack.mean(axis=0)
        flow_t = flow_to_image_tensor(flow_mean)  # 2xHxW
        flow_t = resize_tensor_image(flow_t, self.cfg.out_size)  # 2x64x64
        return flow_t


# ============================================================
# 8) Training / Eval helpers
# ============================================================
def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        rgb_app = batch["rgb_app"].to(device)
        rgb_flow = batch["rgb_flow"].to(device)
        depth_sfi = batch["depth_sfi"].to(device)
        imu_img = batch["imu_img"].to(device)
        y = batch["label"].to(device)

        logits = model(rgb_app, rgb_flow, depth_sfi, imu_img)
        loss = F.cross_entropy(logits, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        rgb_app = batch["rgb_app"].to(device)
        rgb_flow = batch["rgb_flow"].to(device)
        depth_sfi = batch["depth_sfi"].to(device)
        imu_img = batch["imu_img"].to(device)
        y = batch["label"].to(device)

        logits = model(rgb_app, rgb_flow, depth_sfi, imu_img)
        loss = F.cross_entropy(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1)


# ============================================================
# 9) Minimal runnable example (replace with your real data loader)
# ============================================================
def _dummy_item(rgb_len=64, depth_len=64, imu_len=200, H=128, W=128, num_classes=10):
    # dummy RGB frames (OpenCV BGR uint8)
    rgb = [(np.random.rand(H, W, 3) * 255).astype(np.uint8) for _ in range(rgb_len)]
    # dummy depth frames float32 in meters-ish
    depth = [(np.random.rand(H, W).astype(np.float32) * 3.0 + 0.2) for _ in range(depth_len)]
    # dummy imu: (N,6)
    imu = (np.random.randn(imu_len, 6).astype(np.float32) * 0.1)
    label = np.random.randint(0, num_classes)
    return {"rgb_frames": rgb, "depth_frames": depth, "imu_stream": imu, "label": int(label)}


def main_demo():
    if cv2 is None:
        print("cv2 not installed; optical flow will fail. Install: pip install opencv-python")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = WindowConfig(window_seconds=1.04, overlap=0.5, rgb_fps=30, depth_fps=30, imu_hz=50, out_size=64)
    depth_cfg = DepthConfig(use_roi_mask=False, d_min=0.2, d_max=4.0, gamma=1.0)

    items = [_dummy_item(num_classes=6) for _ in range(8)]
    ds = MultimodalWindowDataset(items, cfg, depth_cfg)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = MultimodalHARModel(feat_dim=256, num_classes=6).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(2):
        tr_loss, tr_acc = train_one_epoch(model, dl, optim, device)
        va_loss, va_acc = evaluate(model, dl, device)
        print(f"epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")


if __name__ == "__main__":
    main_demo()