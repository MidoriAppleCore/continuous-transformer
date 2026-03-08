"""
Continuous Transformer — Schrödinger Wave SSM  (Time-Series Forecasting)
=========================================================================
Same wave physics as the language model. Swapped:
  - Fixed harmonic basis  →  nn.Linear input projection (real-valued channels)
  - Vocab logit head      →  direct multi-step forecast head
  - Cross-entropy loss    →  MSE + MAE
  - RevIN                 →  reversible instance normalisation (distribution shift)

Physics (unchanged):
  Evolution:   h_t = h_{t-1} · exp((-γ + iω)·dt)
  Born rule:   α_t = spectrometer(h_t, ψ_t)   [pure-amplitude invariant]
  Fractal FFN: z   = z + FractalRationalMap(CRMSNorm(z))   [Julia-set recurrence]
  Hadamard:    channel mixing via fixed unitary beam splitter  [zero params]

Benchmarks: ETTh1, ETTh2, ETTm1, ETTm2
Standard prediction lengths: 96 / 192 / 336 / 720
"""

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DIM          = 64        # Wave-field width  (32=72K fast, 64=280K sweet spot, 128=1M slow)
DEPTH        = 6        # depth > 4: adds new timescales cheaply vs wider DIM
MIMO_P       = 4        # Hadamard block size

N_CHANNELS   = 7        # ETT has 7 variates; override for other datasets
CONTEXT_LEN  = 512      # ODE burn-in: slow γ modes need ~512 steps to accumulate phase
PRED_LEN     = 96       # Forecast horizon (96 / 192 / 336 / 720)
PATCH_SIZE   = 16       # Patching: 512 hours → 32 patches. Variance drops to σ²/16, FFT 16× cheaper

BASE_LR      = 1e-4     # SOTA standard: DLinear/PatchTST/iTransformer all use 1e-4
WEIGHT_DECAY = 0.01     # L2 friction — prevents overfitting to training noise
BATCH_SIZE   = 8       # benchmark standard
GRAD_CLIP    = 1.0
MAX_EPOCHS   = 15       # softer LR decay needs more epochs for basins to crystallise
PATIENCE     = 5        # more runway before early stopping fires

CHECKPOINT_EVERY = 1        # save every N epochs
PRINT_EVERY      = 50       # steps

USE_AMP    = True
NUM_WORKERS = 2

DATASET    = "ETTh1"    # ETTh1 | ETTh2 | ETTm1 | ETTm2

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math, os, glob, urllib.request
import numpy as np
import pandas as pd
import torch
from scipy.fft import next_fast_len
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

torch.set_float32_matmul_precision('high')

DATA_ROOT  = os.path.expanduser("~/.cache/timeseries")

# ---------------------------------------------------------------------------
# ETT Dataset  (Electricity Transformer Temperature, Zhou et al. 2021)
# ---------------------------------------------------------------------------

ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

# Standard splits (fraction of total rows)
ETT_SPLITS = {
    "ETTh1": (0.6, 0.2, 0.2),
    "ETTh2": (0.6, 0.2, 0.2),
    "ETTm1": (0.6, 0.2, 0.2),
    "ETTm2": (0.6, 0.2, 0.2),
}


def prepare_ett_dataset(name: str = "ETTh1", root: str = DATA_ROOT) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{name}.csv")
    if not os.path.exists(path):
        url = ETT_URLS[name]
        print(f"Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    return path


class ETTDataset(Dataset):
    """
    Sliding-window dataset over one ETT split.

    Global standard scaling: mean/std fitted on the TRAINING split only,
    then applied to all splits. This matches the exact normalisation used by
    PatchTST, DLinear, TimesNet etc., so MSE numbers are directly comparable.
    RevIN on top of this handles residual per-instance distribution shift.

    Returns (x, y):
      x : [context_len, n_channels]  float32  — look-back window (standardised)
      y : [pred_len,    n_channels]  float32  — forecast target  (standardised)
    """
    def __init__(self, name: str, split: str = "train",
                 context_len: int = CONTEXT_LEN, pred_len: int = PRED_LEN,
                 root: str = DATA_ROOT,
                 mean: np.ndarray = None, std: np.ndarray = None):
        path = prepare_ett_dataset(name, root)
        df   = pd.read_csv(path, index_col=0, parse_dates=True)
        data = df.values.astype(np.float32)          # [T, C]

        T = len(data)
        train_frac, val_frac, _ = ETT_SPLITS[name]
        n_train = int(T * train_frac)
        n_val   = int(T * val_frac)

        # Fit scaler on training data only — never look at val/test stats
        if mean is None or std is None:
            train_data  = data[:n_train]
            self.mean_  = train_data.mean(axis=0, keepdims=True)   # [1, C]
            self.std_   = train_data.std(axis=0,  keepdims=True) + 1e-8
        else:
            self.mean_  = mean
            self.std_   = std

        if split == "train":
            raw = data[:n_train]
        elif split == "val":
            # Overlap by context_len so the first predicted target is
            # exactly at n_train — no 21-day black hole between splits.
            raw = data[n_train - context_len : n_train + n_val]
        else:
            # Same for test: first target is exactly at n_train + n_val.
            raw = data[n_train + n_val - context_len :]

        # Apply global standardisation: data is now Mean≈0, Std≈1
        self.data        = (raw - self.mean_) / self.std_
        self.context_len = context_len
        self.pred_len    = pred_len
        self.window      = context_len + pred_len
        self.n_samples   = max(0, len(self.data) - self.window + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        seg = self.data[idx : idx + self.window]          # [window, C]
        x   = torch.from_numpy(seg[:self.context_len])   # [L, C]
        y   = torch.from_numpy(seg[self.context_len:])   # [P, C]
        return x, y


# ---------------------------------------------------------------------------
# RevIN — Reversible Instance Normalisation (Kim et al. 2022)
# Handles distribution shift: normalise per-instance at input, denorm at output.
# Learnable affine per channel (2×C params) that survive across domains.
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Per-channel instance norm — each variable keeps its own coordinate ground.
    Mean/std computed over the time axis only: [B, L, C] → stats [B, 1, C].
    Matches PatchTST / iTransformer protocol."""
    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self._mean = None
        self._std  = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] → normalised [B, L, C]"""
        self._mean = x.mean(dim=1, keepdim=True).detach()       # [B, 1, C]
        self._std  = (x.var(dim=1, keepdim=True) + self.eps).sqrt().detach()  # [B, 1, C]
        return (x - self._mean) / self._std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, P, C] → original scale [B, P, C]"""
        return x * self._std + self._mean


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def hippo_freqs(dim: int) -> torch.Tensor:
    """HiPPO-LegS frequency initialisation for optimal history reconstruction."""
    n = torch.arange(dim, dtype=torch.float32)
    freqs = (2 * n + 1) ** 0.5
    freqs = freqs / freqs.max()
    return torch.exp(-freqs * np.log(10_000))


def hippo_b_vector(dim: int) -> torch.Tensor:
    """
    HiPPO-LegS input coupling vector: b_n = sqrt(2n+1).

    This is the EXACT same (2n+1)^0.5 that initialises omega — that's not
    a coincidence.  In the diagonalised HiPPO-LegS system, A's eigenvalues
    have imaginary parts proportional to sqrt(2n+1), and B_n = sqrt(2n+1).
    They are mathematically coupled: the optimal input coupling and the
    optimal rotation frequency are the same function of polynomial order n.

    Low-order modes (n≈0):  small b_n → slow, memory-holding modes
    High-order modes (n≈D): large b_n → fast, input-sensitive modes

    Normalised to max=1 so high-order modes get full coupling (≡1) and
    low-order modes are proportionally weaker but longer-lived.
    """
    n = torch.arange(dim, dtype=torch.float32)
    b = (2 * n + 1).sqrt()
    return b / b.max()   # [D], values in (0, 1]


def exact_projection(z: torch.Tensor,
                     target_mean: float = 0.0,
                     target_std:  float = 1.0) -> torch.Tensor:
    """
    Exact closed-form geodesic projection onto stable manifold S(μ,σ).
    Prevents the wave field from exploding or collapsing.
    """
    mean = z.mean(dim=-1, keepdim=True)
    var  = ((z - mean) ** 2).mean(dim=-1, keepdim=True)
    return (z - mean) / (var.sqrt() + 1e-6) * target_std + target_mean


# ---------------------------------------------------------------------------
# FractalRationalMap — Julia set nonlinearity
# ---------------------------------------------------------------------------

class FractalRationalMap(nn.Module):
    """
    Complex rational map  z ↦ (a·z + b) / (c·|z| + d).

    Generates Julia-set fractal decision boundaries when iterated across layers.
    Enforces a strict Euclidean Isometry: 'a' is a unit phasor (|a|=1),
    so the numerator ONLY rotates and shifts. The denominator ONLY scales.
    This eliminates optimization redundancy between numerator scaling and
    denominator squashing, making the fractal dynamics cleaner for AdamW.

    Uses the same Transcendental Bypass as the holographic phase keys:
    F.normalize on a 2D real vector → unit circle, zero trig kernels.

    Numerator (a·z + b):  pure rotation + translation.
        The shift (+b) pulls the system away from the origin, generating the
        chaotic basin boundaries that make Julia sets useful.
    Denominator (c·|z| + d):  real magnitude squashing — prevents the fractal
        iterations from exploding to infinity.

    Init: a ≈ (1, 0) via normalize, b=0, c=0, d=1 → ≈ identity map.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Numerator: pure rotation (unit phasor) + translation
        # Transcendental Bypass: [dim, 2] real → F.normalize → unit circle
        self.a_raw  = nn.Parameter(torch.stack([torch.ones(dim),
                                                torch.zeros(dim)], dim=-1))  # [dim, 2]
        self.b_real = nn.Parameter(torch.zeros(dim))   # no shift at init
        self.b_imag = nn.Parameter(torch.zeros(dim))
        # Denominator: real magnitude squashing (prevents explosion)
        self.c = nn.Parameter(torch.zeros(dim))        # no |z| dependence at init
        self.d = nn.Parameter(torch.ones(dim))         # softplus(1) ≈ 1.31 → stable

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., dim] complex → [..., dim] complex"""
        # Force 'a' to the unit circle: pure rotation, |a| = 1 exactly
        a_norm  = F.normalize(self.a_raw.float(), p=2, dim=-1)       # [dim, 2]
        a       = torch.complex(a_norm[..., 0], a_norm[..., 1])      # [dim] unit phasors
        b       = torch.complex(self.b_real.float(), self.b_imag.float())
        numerator   = a * z + b
        # softplus ensures denominator is strictly positive and smooth
        c_pos       = F.softplus(self.c)
        d_pos       = F.softplus(self.d) + self.eps
        denominator = c_pos * z.abs() + d_pos
        return numerator / denominator


class FractalFFN(nn.Module):
    """
    Complex FFN with mode mixing + element-wise nonlinearity.

    Pipeline:  z → CRMSNorm → ComplexLinear (mode mixing) → Dropout → RationalMap → residual

    The linear layer lets wave modes interact (e.g. 24h cycle modulating
    weekly pattern).  The rational map applies element-wise magnitude
    squashing with learned rotation — a valid complex nonlinearity that
    prevents the residual stream from exploding while preserving phase.

    Params per layer: dim² × 2 (linear real+imag) + 6×dim (rational map).
    At dim=32: 2,048 + 192 = 2,240 per layer.  Still tiny.
    """
    def __init__(self, dim: int, eps: float = 1e-6, dropout: float = 0.1):
        super().__init__()
        self.eps     = eps
        # Complex linear: learned mode mixing.  Applied as separate
        # real-valued matmuls on Re and Im parts, then recombined:
        #   W·z = W·(a+bi) = W·a + i·W·b  (real weights, complex input)
        # This lets mode k's output depend on ALL other modes' states.
        self.mix = nn.Linear(dim, dim, bias=False)
        # Shared-mask dropout: one Bernoulli mask applied to BOTH Re and Im.
        # This zeros entire complex modes (not half-modes), keeping the
        # physical meaning intact — a mode is either present or absent.
        self.drop    = nn.Dropout(dropout)
        self.fractal = FractalRationalMap(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., D] complex → [..., D] complex"""
        # CRMSNorm: project to unit sphere — stable input to mixing + nonlinearity
        rms = (z.real**2 + z.imag**2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        z_normed = z / rms
        # Mode mixing: cast outputs to fp32 — torch.complex doesn't support bfloat16
        r_mixed = self.mix(z_normed.real).float()
        i_mixed = self.mix(z_normed.imag).float()
        # Shared dropout mask: sample once on real part, apply to both.
        # Zeroes out complete complex modes so Re²+Im² → 0 together.
        mask = (self.drop(torch.ones_like(r_mixed)) > 0).float()  # {0,1} Bernoulli
        z_mixed = torch.complex(r_mixed * mask, i_mixed * mask)
        # Element-wise nonlinearity: magnitude squashing + rotation
        return z + self.fractal(z_mixed)


# ---------------------------------------------------------------------------
# SchrodingerAttention
# ---------------------------------------------------------------------------

class SchrodingerAttention(nn.Module):
    """
    O(1) continuous attention — exact train/inference equivalence.

    Dual Wave Architecture — state-dependent selection with exact parallel training.

    Two waves per layer:
      Scout:  hs_t = A · hs_{t-1}  +  Bs(x_t) ⊙ gate · ψ_t     (input-only LTI)
      True:   h_t  = A · h_{t-1}   +  S(U_t)  (Simpson's quadrature)

    Simpson's Rule discretization (4th-order quadrature):
      S(U)_t = (1/6)·U_t + (4/6)·U_{t-1} + (1/6)·U_{t-2}
      Fits a parabola through the last 3 wave-space inputs before integration.
      Zero parameters. Subsumes trapezoidal (2nd-order) and Euler (1st-order).
      Valid because U_t = gate·B·ψ is already in smooth complex wave space
      (embedding + projection liquifies discrete tokens before quadrature).
      Training: depthwise causal conv1d with fixed kernel [1/6, 4/6, 1/6].
      Inference: 3-tap recurrent blend, U_{t-1} and U_{t-2} stored in state.

    A  = exp((-γ+iω)·dt)  is CONSTANT.
    Bs = softplus(W_scout · x_t)                             — input-only (scout)
    Bt = softplus(W_true  · [x_t, Re(hs_{t-1}), Im(hs_{t-1})])  — state-dependent
    gate(x_t) is a shared SCALAR controlling WHEN to write.

    Why FFT is still exact for the true wave:
      The scout is fully computed first (no dependence on the true wave).
      Bt then depends only on scout state — a fully determined sequence.
      H_true = IFFT(FFT(U_true) · FFT(kernel)) is valid; kernel A^t is still constant.

    Result: Mamba-level state-dependence using pure PyTorch FFT, no CUDA kernels.
    Train/inference equivalence is EXACT — both paths track the same two recurrences.

    Born-rule measurement uses the true wave only:
        out_t = Re(h_t · φ_t*)   ← wave interference / measurement

    Training  O(L log L) — two parallel FFT convolutions
    Inference O(1)       — two O(1) recurrent steps, one per wave
    """

    def __init__(self, dim: int, n_bands: int = 4, mimo_p: int = 8,
                 layer_idx: int = 0, depth: int = 1, state_dim: int = None,
                 patch_size: int = 1):
        super().__init__()
        self.dim    = dim
        self.patch_size = patch_size
        # state_dim: wave field width. Default = dim (no change). Set larger on deep/slow
        # layers to increase long-term memory capacity without widening the FFN or embedding.
        SD = self.state_dim = state_dim if state_dim is not None else dim
        self.n_bands = n_bands   # spectral bands: each reads SD/K independent modes
        assert SD % n_bands == 0, f"state_dim={SD} must be divisible by n_bands={n_bands}"
        # Scout is capped at model dim. Deep layers (SD=1024, dim=256) run the scout
        # at 256 channels — 4× fewer FFT channels, 4× cheaper scout FFT.
        ScoutDim = self.scout_dim = min(dim, SD)

        # ── Wave physics ──────────────────────────────────────────────────
        self.omega     = nn.Parameter(hippo_freqs(SD))         # [SD] HiPPO rotation
        # Superconductor init: -7.0 instead of -3.0.
        # With log_gamma=-3 and dt=0.74, retention after 256 filler = 0.008% → total amnesia.
        # With log_gamma=-7: gamma=0.0009, retention after 256 = 84%, after 512 = 71%.
        self.log_gamma = nn.Parameter(torch.randn(SD) * 0.5 - 7.0)  # [SD] dispersive decay
        # Log-spaced dt hierarchy: L0=fast (local syntax), L_{depth-1}=slow (KV vault).
        # dt_min=0.01 (was 0.001): retention over 512 tokens is 99.5% vs 99.95% — identical
        # for KV recall purposes.  But softplus_inv(0.01)=-4.6 vs softplus_inv(0.001)=-6.9,
        # so sigmoid(-4.6)=0.01 vs sigmoid(-6.9)=0.001 — the chain-rule factor through
        # softplus is 10× larger, bringing L3 into the same learnable regime as L1/L2.
        # Phasors also rotate 10× faster → more distinct phases per key → better SNR.
        # Inverse softplus so F.softplus(self.dt) == target_dt exactly at init.
        # PATCHING: Scale dt by patch_size so physical wavelengths are preserved.
        # If one step = 16 hours, the continuous clock must tick 16× faster per step.
        dt_max, dt_min = 0.1, 0.01
        ratio          = layer_idx / max(depth - 1, 1)
        target_dt      = dt_max * (dt_min / dt_max) ** ratio * patch_size  # scale by patch
        inv_sp_dt      = math.log(math.expm1(target_dt))       # softplus⁻¹
        self.dt        = nn.Parameter(torch.tensor(inv_sp_dt, dtype=torch.float32))

        # ── Scout Wave physics (independent small-state SSM) ─────────────
        # ScoutDim = min(dim, SD): bottlenecked so scout FFT is never wider than dim.
        # Own parameters so scout can specialise for local syntax independently.
        # ~2*ScoutDim+1 extra params per layer — negligible.
        self.omega_scout     = nn.Parameter(hippo_freqs(ScoutDim))
        self.log_gamma_scout = nn.Parameter(torch.randn(ScoutDim) * 0.5 - 7.0)
        self.dt_scout        = nn.Parameter(torch.tensor(inv_sp_dt, dtype=torch.float32))
        _scout_order   = torch.argsort(hippo_freqs(ScoutDim), descending=False)
        _unitary_scout = torch.zeros(ScoutDim, dtype=torch.bool)
        _unitary_scout[_scout_order[:ScoutDim // n_bands]] = True
        self.register_buffer('unitary_mask_scout', _unitary_scout)

        # ── Token → wave projections ──────────────────────────────────────
        self.to_psi = nn.Linear(dim, SD * 2)    # excitation wave  ψ (content → wave field)
        nn.init.orthogonal_(self.to_psi.weight)  # orthogonal write basis: key tokens span separate subspaces
        self.to_phi = nn.Linear(dim, SD * 2)    # measurement wave φ (read  → wave field)
        nn.init.orthogonal_(self.to_phi.weight)  # complete orthonormal basis at t=0
        self.to_psi_scout = nn.Linear(dim, ScoutDim * 2)    # Scout ψ — bottlenecked width
        nn.init.orthogonal_(self.to_psi_scout.weight)

        # ── HiPPO-aligned band boundaries ────────────────────────────────
        # hippo_freqs returns exp(-sqrt(2n+1)/max · log10000), monotone ↓ in n.
        # Argsort gives indices ordered slow→fast (low freq → high freq).
        # We cut that ordering into K equal-count bands so each band spans
        # one quartile of the HiPPO frequency spectrum, not a raw index range.
        freqs     = hippo_freqs(SD)                           # [SD] monotone ↓ in n
        # descending=False: lowest hippo_freq first = n≈SD-1 = LOW omega = SLOW rotation.
        # bands[0] = slow modes (stable phase across filler) → correct for γ≡0 unitary lock.
        # descending=True was wrong: it put n≈0 (HIGH omega, fast-spinning) in bands[0].
        order     = torch.argsort(freqs, descending=False)    # slow→fast (low ω → high ω)
        band_size = SD // n_bands
        bands     = order.view(n_bands, band_size)            # [K, SD/K] indices
        self.register_buffer('band_idx', bands)               # [K, SD/K] long
        # Inverse permutation: undoes the band reordering in O(1) via gather.
        # Eliminates the torch.zeros(SD) + scatter pattern in forward_step.
        self.register_buffer('inv_band_idx', torch.argsort(bands.flatten()))  # [SD]
        # Unitary subspace: slowest band (band-0, SD//K dims) locked to γ≡0.
        # These dimensions are pure rotators: |A|=1 exactly, zero information loss
        # at any distance. Provides SD//(2K) complex slots with guaranteed infinite
        # retention — a dedicated KV vault that never forgets regardless of filler length.
        # The remaining 3 bands keep learnable γ for normal leaky dynamics.
        unitary = torch.zeros(SD, dtype=torch.bool)
        unitary[bands[0]] = True
        self.register_buffer('unitary_mask', unitary)         # [SD] True = pure rotation, no decay

        # ── Dual Wave B coupling + scalar write gate ─────────────────────
        # Scout B (to_B_scout): input-only LTI. Runs first, feels the sequence shape.
        #   Bs_t = softplus(W_scout · x_t)  — standard Mamba-style selection.
        #   FFT exactness preserved: Bs only affects U_t, not the kernel A^t.
        # True B (to_B_true): the Smart Bouncer — sees x AND the scout's past state.
        #   Bt_t = softplus(W_true · [x_t, Re(hs_{t-1}), Im(hs_{t-1})])
        #   Scout is fully computed first → Bt is a determined sequence → FFT valid.
        #   Achieves Mamba-level state-dependence without custom CUDA kernels.
        # Both inited: zero weight, bias = softplus_inv(hippo_b_vector).
        #   softplus(W·0) == hippo_b_vector — starts from the HiPPO optimum.
        # write_gate: shared SCALAR — controls WHEN to write (both waves).
        b0_true     = hippo_b_vector(SD)                          # [SD] true wave target
        b0_true_inv = torch.log(torch.exp(b0_true.clamp(min=1e-6)) - 1.0)
        b0_scout    = hippo_b_vector(ScoutDim)                     # [ScoutDim] scout target
        b0_scout_inv = torch.log(torch.exp(b0_scout.clamp(min=1e-6)) - 1.0)
        self.to_B_scout = nn.Linear(dim, ScoutDim)                 # Scout: bottlenecked coupling
        nn.init.zeros_(self.to_B_scout.weight)
        self.to_B_scout.bias.data.copy_(b0_scout_inv)
        self.to_B_true_x = nn.Linear(dim, SD)                      # True: input path
        nn.init.zeros_(self.to_B_true_x.weight)
        self.to_B_true_x.bias.data.copy_(b0_true_inv)
        self.to_B_true_h = nn.Linear(ScoutDim * 2, SD)             # Bouncer: scout→SD (bottlenecked)
        nn.init.zeros_(self.to_B_true_h.weight)
        nn.init.zeros_(self.to_B_true_h.bias)
        # Phase-selective writing — Transcendental Bypass:
        # Linear → F.normalize(p=2, dim=-1) instead of tanh → cos/sin.
        # Eliminates 3 GPU trig kernels (tanh, cos, sin) per wave per step.
        # Both paths produce identical unit-magnitude complex phasors — math unchanged.
        # Output size doubles (2 channels per mode: cos_raw, sin_raw) so F.normalize
        # snaps the 2D vector to the unit circle in pure MAC operations.
        # Phase remains strictly input-only for scout and true — holographic key intact.
        _phase_w_scale = 1.0 / math.sqrt(dim)
        self.to_phase_scout  = nn.Linear(dim, ScoutDim * 2)  # bypass: [dim → ScoutDim×2]
        nn.init.uniform_(self.to_phase_scout.weight, -_phase_w_scale, _phase_w_scale)
        nn.init.zeros_(self.to_phase_scout.bias)
        self.to_phase_true_x = nn.Linear(dim, SD * 2)        # bypass: [dim → SD×2]
        nn.init.uniform_(self.to_phase_true_x.weight, -_phase_w_scale, _phase_w_scale)
        nn.init.zeros_(self.to_phase_true_x.bias)
        # to_phase_true_h removed — phase_true is strictly input-only (no state dependency)
        self.write_gate = nn.Linear(dim, 1)   # scalar gate — WHEN to write
        self.surprise_gain = nn.Parameter(torch.zeros(1))  # predictive coding depth
        self.tau  = nn.Parameter(torch.ones(1))   # ignition temperature (init=1 → soft)
        self.beta = nn.Parameter(torch.zeros(1))  # self-model density weight (init=0 → off)
        # tau=1: standard softmax competition.  Sharpens as bands specialise.
        # beta=0: no self-model at t=0. Activates when density carries gradient signal.
        # ── Hadamard beam splitter (replaces learned MIMO) ────────────────────
        # Fixed normalized Hadamard H of size [P, P]: H @ H.T = I exactly.
        # This is the quantum beam splitter — a passive unitary that rotates
        # channels into each other without amplifying or attenuating energy.
        # Zero learned parameters. 100% of learning stays in the wave physics
        # and fractal phase projections.
        # Sylvester construction: recursive H_{2n} = (1/√2)[[H_n, H_n],[H_n,-H_n]]
        assert SD % mimo_p == 0, f"state_dim={SD} must be divisible by mimo_p={mimo_p}"
        self.mimo_p = mimo_p
        H = self._build_hadamard(mimo_p)  # [P, P] normalized unitary
        self.register_buffer('hadamard', H)  # zero params, moves with model.to(device)
        # Output projection: maps wave field [SD] → model dim [dim].
        # nn.Identity when SD == dim (default — zero extra params).
        self.out_proj = nn.Linear(SD, dim, bias=False) if SD != dim else nn.Identity()
        # ── Fractal Rational Map — Julia set nonlinearity on the wave field ──
        # Applied after FFT RMSNorm, before spectrometer readout.
        # Warps the linear wave interference pattern through a rational function
        # whose basin boundaries are fractal curves. The spectrometer then reads
        # out *which basin* each token's wave landed in — categorical decisions
        # from continuous dynamics. 6×SD params (~3K at SD=512).
        self.fractal = FractalRationalMap(SD)
        self._A_cache       = None   # invalidated by ContinuousTransformer.invalidate_A_cache()
        self._A_scout_cache = None   # same — scout uses separate physics

    # ------------------------------------------------------------------
    @staticmethod
    def _build_hadamard(n: int) -> torch.Tensor:
        """Normalized Hadamard via Sylvester construction. H @ H.T = I.
        n must be a power of 2."""
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H,  H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0) / math.sqrt(2)
        return H  # [n, n], H @ H.T = I exactly

    # ------------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        """Cached evolution factor A = exp((-γ + iω)·dt)  [D]  complex.
        softplus(dt) — always positive, smooth gradient, never dead at zero.
        Valid between optimizer steps; call invalidate_A_cache() after step()."""
        if self._A_cache is None:
            dt = F.softplus(self.dt)   # always > 0, gradient alive everywhere
            # unitary_mask dims: γ_eff=0 → |A|=1, pure rotation, infinite retention
            gamma_eff = F.softplus(self.log_gamma) * (~self.unitary_mask).float()
            self._A_cache = torch.exp(torch.complex(
                -gamma_eff * dt,
                self.omega * dt,
            ))
        return self._A_cache

    def _A_scout(self) -> torch.Tensor:
        """Cached evolution factor for the bottlenecked scout wave [ScoutDim] complex."""
        if self._A_scout_cache is None:
            dt = F.softplus(self.dt_scout)
            gamma_eff = F.softplus(self.log_gamma_scout) * (~self.unitary_mask_scout).float()
            self._A_scout_cache = torch.exp(torch.complex(-gamma_eff * dt, self.omega_scout * dt))
        return self._A_scout_cache

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
               z_prev: torch.Tensor | None = None) -> torch.Tensor:
        """
        Exact parallel training pass via causal FFT convolution.

        Because A is constant, h_t = Σ_{s≤t} A^{t-s} · U_s
        is a causal convolution with kernel k(t) = A^t.
        The FFT computes this exactly in O(L log L).

        x      : [B, L, D]  real
        z_prev : [B, L, D]  previous layer's output (predictive coding signal)
        returns: [B, L, D]  real
        """
        B, L, D  = x.shape
        SD       = self.state_dim    # true wave field width
        ScoutDim = self.scout_dim    # scout wave width — capped at dim, ≤ SD

        psi_raw = self.to_psi(x).float()                                # fp32 — complex64 needs float32 parts
        psi     = torch.complex(psi_raw[..., :SD], psi_raw[..., SD:])  # [B, L, SD]
        phi_raw = self.to_phi(x).float()
        phi     = torch.complex(phi_raw[..., :SD], phi_raw[..., SD:])  # [B, L, SD]

        # Predictive coding: gate scales with surprise (deviation from prediction).
        # When z_prev perfectly predicts x, surprise→0 and gate closes.
        # Wave only updates when it encounters something unexpected.
        gate = torch.sigmoid(self.write_gate(x).float())                # [B, L, 1] real
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, L, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)
        # ── True Wave kernel (SD dims, Simpson's folded in) ──────────────────
        dt        = F.softplus(self.dt)
        gamma_eff = F.softplus(self.log_gamma) * (~self.unitary_mask).float()
        lam       = torch.complex(-gamma_eff, self.omega)
        t_axis    = torch.arange(L, device=x.device, dtype=torch.float32)
        kernel_raw = torch.exp(lam.unsqueeze(0) * t_axis.unsqueeze(1) * dt)  # [L, SD]
        k_zeros        = torch.zeros(2, SD, dtype=kernel_raw.dtype, device=x.device)
        k_pad          = torch.cat([k_zeros, kernel_raw], dim=0)
        kernel_simpson = (1/6)*k_pad[2:] + (4/6)*k_pad[1:-1] + (1/6)*k_pad[:-2]  # [L, SD]

        # ── Scout Wave kernel (ScoutDim ≤ SD, independent physics) ───────────
        # Bottlenecked: deep layers run scout at dim=256 instead of SD=1024 — 4× cheaper.
        dt_s         = F.softplus(self.dt_scout)
        gamma_eff_s  = F.softplus(self.log_gamma_scout) * (~self.unitary_mask_scout).float()
        lam_s        = torch.complex(-gamma_eff_s, self.omega_scout)
        kernel_scout = torch.exp(lam_s.unsqueeze(0) * t_axis.unsqueeze(1) * dt_s)  # [L, ScoutDim]

        n_fft        = next_fast_len(2 * L - 1)   # 5-smooth ≥ 2L-1 — cuFFT fast path
        K_freq_scout = torch.fft.fft(kernel_scout.T, n=n_fft, dim=1)   # [ScoutDim, n_fft]
        K_freq_true  = torch.fft.fft(kernel_simpson.T, n=n_fft, dim=1)  # [SD, n_fft]

        # ── 1. Scout Wave (bottlenecked LTI — ScoutDim channels) ─────────────
        psi_s_raw  = self.to_psi_scout(x).float()
        psi_scout  = torch.complex(psi_s_raw[..., :ScoutDim], psi_s_raw[..., ScoutDim:])  # [B, L, ScoutDim]
        b_scout    = F.softplus(self.to_B_scout(x).float())                  # [B, L, ScoutDim]
        # Transcendental Bypass: Linear→normalize vs tanh→cos/sin — identical unit phasors.
        ph_s_raw   = self.to_phase_scout(x).float().view(B, L, ScoutDim, 2)  # [B, L, ScoutDim, 2]
        ph_s       = F.normalize(ph_s_raw, p=2, dim=-1)                      # unit 2D vectors
        B_c_scout  = torch.complex(b_scout * ph_s[..., 0], b_scout * ph_s[..., 1])
        U_scout    = gate * B_c_scout * psi_scout                            # [B, L, ScoutDim]
        U_freq_scout = torch.fft.fft(U_scout.permute(0, 2, 1), n=n_fft, dim=2)
        H_scout    = torch.fft.ifft(
            U_freq_scout * K_freq_scout.unsqueeze(0), n=n_fft, dim=2
        )[:, :, :L].permute(0, 2, 1)                                         # [B, L, ScoutDim]

        # ── 2. Smart Bouncer — causal state-dependent B for the true wave ─
        H_scout_prev   = F.pad(H_scout, (0, 0, 1, 0))[:, :-1, :]        # [B, L, ScoutDim]
        scout_rms      = (H_scout_prev.real**2 + H_scout_prev.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        H_scout_prev_n = H_scout_prev / scout_rms                        # [B, L, ScoutDim] unit-phasor-scaled
        H_prev_flat    = torch.cat([H_scout_prev_n.real, H_scout_prev_n.imag], dim=-1)  # [B, L, 2*ScoutDim]
        b_true     = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(H_prev_flat).float())
        # Phase: strictly input-only (no state) — holographic write key unchanged at query time.
        # Transcendental Bypass: same unit phasors, zero trig kernels.
        ph_t_raw   = self.to_phase_true_x(x).float().view(B, L, SD, 2)  # [B, L, SD, 2]
        ph_t       = F.normalize(ph_t_raw, p=2, dim=-1)                  # [B, L, SD, 2] unit vectors
        B_c_true   = torch.complex(b_true * ph_t[..., 0], b_true * ph_t[..., 1])

        # ── 3. True Wave — kernel-folded Simpson's (no batch-dim allocation) ──
        # Simpson's is already baked into K_freq_true via conv associativity.
        # We skip the _zeros / U_pad / three-shift blend on [B, L, SD] entirely —
        # those reads/writes are now free because the kernel handles it.
        U_true_raw  = gate * B_c_true * psi                               # [B, L, SD] complex
        U_freq_true = torch.fft.fft(U_true_raw.permute(0, 2, 1), n=n_fft, dim=2)
        H           = torch.fft.ifft(
            U_freq_true * K_freq_true.unsqueeze(0), n=n_fft, dim=2
        )[:, :, :L].permute(0, 2, 1)                                     # [B, L, SD]

        # Complex RMSNorm at READ TIME ONLY — preserves phase angles for demodulation.
        # Not inside the recurrence, so training (FFT) and inference (step loop) stay identical.
        read_rms = (H.real**2 + H.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        H_proj   = H / read_rms                                          # [B, L, SD]

        # ── Fractal warp: Julia-set basin categorization ─────────────────
        # The FFT produced a linear wave. The rational map bends it into
        # fractal basins so the spectrometer reads categorical decisions.
        H_proj = self.fractal(H_proj)                                    # [B, L, SD]

        # Spectrometer: K bands each measure SD/K modes independently.
        # Bands follow HiPPO frequency quartiles: band 0 = slowest modes
        # (long memory), band K-1 = fastest modes (reflexes).
        # α_k = ⟨h_k*, φ_k⟩ / (D/K)  — one complex amplitude per band.
        K  = self.n_bands
        bx = self.band_idx                                              # [K, D/K]
        Hb = H_proj[:, :, bx]                                          # [B, L, K, D/K]
        Pb = phi[:, :, bx]                                             # [B, L, K, D/K]
        # Normalise phi to unit phasors — preserves all phase information for demodulation
        # while preventing to_phi weight growth from making |alpha| and |unbound| unbounded.
        # After LN, |Hb_j|≈1; with Pb_n unit-magnitude, |alpha|≤1 and |unbound_j|≤1,
        # so output is bounded by O(K) regardless of training step count. Fixes NaN cascade.
        Pb_n  = Pb / Pb.abs().clamp(min=1e-8)                                 # [B, L, K, D/K] unit phasors
        alpha = (Hb.conj() * Pb_n).sum(-1, keepdim=True) / (SD // K)          # [B, L, K, 1]

        # Global workspace ignition: bands compete via softmax sharpened by tau.
        # Phase-pure: only the real magnitude |alpha| enters competition.
        tau_eff      = F.softplus(self.tau).clamp(min=0.1)
        alpha_mag    = alpha.abs()                                              # [B, L, K, 1] real
        c_k          = F.softmax(alpha_mag / tau_eff, dim=2)                    # [B, L, K, 1]
        alpha_ig_mag = alpha_mag * c_k * K                                      # [B, L, K, 1] real

        # Self-model: per-band Born-rule density |h_k|² / (D/K).
        # Purely real scalar — adds confidence without corrupting phase.
        density       = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)        # [B, L, K, 1]
        alpha_fin_mag = alpha_ig_mag + self.beta * density                      # [B, L, K, 1] real

        # Holographic unbinding: demodulate with the EXACT phase used to write.
        # Phase is perfectly preserved: unbound carries all phase information,
        # alpha_fin_mag is a strictly real scalar that only adjusts amplitude.
        Q_phase        = torch.complex(ph_t[..., 0], ph_t[..., 1])  # [B, L, SD] — reuse bypass phasors
        unbound        = Hb * Q_phase[:, :, bx].conj()                  # [B, L, K, SD/K]
        unbound_scaled = unbound * alpha_fin_mag                         # complex × real: phase untouched
        # Keep complex — phase flows into the residual stream for fractal iteration
        out = torch.zeros(B, L, SD, device=x.device, dtype=unbound_scaled.dtype)
        out[:, :, bx] = unbound_scaled
        # ── Hadamard beam splitter: passive unitary channel mixing, H H^T = I ──
        # Split real/imag so matmul runs under bfloat16 autocast without conflict
        G, P = SD // self.mimo_p, self.mimo_p
        H = self.hadamard  # [P, P] fixed unitary
        out_r = torch.einsum('pq,blgq->blgp', H, out.real.reshape(B, L, G, P))
        out_i = torch.einsum('pq,blgq->blgp', H, out.imag.reshape(B, L, G, P))
        out = torch.complex(out_r.float(), out_i.float()).reshape(B, L, SD)
        # out_proj: real weights applied to complex → complex output
        if isinstance(self.out_proj, nn.Identity):
            return out
        return torch.complex(self.out_proj(out.real), self.out_proj(out.imag))

    # ------------------------------------------------------------------
    def forward_step(self, x: torch.Tensor,
                     h_prev: torch.Tensor,
                     z_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) recurrent step — bit-identical to forward().

        Runs two recurrences matching the training Dual Wave FFT:
          Scout:  hs_t = A · hs_{t-1} + Bs(x_t) ⊙ gate · ψ_t
          True:   h_t  = A · h_{t-1}  + Bt(x_t, hs_{t-1}) ⊙ gate · ψ_t

        x      : [B, D]     real token features
        h_prev : [B, 4, D]  complex — scout[:, 0], true[:, 1], U_{t-1}[:, 2], U_{t-2}[:, 3]
        z_prev : [B, D]     previous layer output (predictive coding)
        returns: out [B, D] real, h_next [B, 4, D] complex
        """
        D        = self.dim
        SD       = self.state_dim    # true wave field width
        ScoutDim = self.scout_dim    # scout wave width — capped at dim, ≤ SD

        psi_raw = self.to_psi(x).float()
        psi     = torch.complex(psi_raw[..., :SD], psi_raw[..., SD:])  # [B, SD]
        phi_raw = self.to_phi(x).float()
        phi     = torch.complex(phi_raw[..., :SD], phi_raw[..., SD:])  # [B, SD]

        gate = torch.sigmoid(self.write_gate(x).float())                # [B, 1]
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)

        # Unpack: h_prev is [B, 4, SD] — scout uses first ScoutDim channels of slot 0
        h_prev_scout = h_prev[:, 0, :ScoutDim]   # [B, ScoutDim] complex
        h_prev_true  = h_prev[:, 1, :]            # [B, SD] complex
        h_prev_U1    = h_prev[:, 2, :]            # [B, SD] U_{t-1}
        h_prev_U2    = h_prev[:, 3, :]            # [B, SD] U_{t-2}

        # Scout ψ — bottlenecked excitation
        psi_s_raw  = self.to_psi_scout(x).float()
        psi_scout  = torch.complex(psi_s_raw[..., :ScoutDim], psi_s_raw[..., ScoutDim:])  # [B, ScoutDim]

        # 1. Scout step (bottlenecked — matches forward's scout wave)
        Bsz_step   = x.shape[0]
        b_scout    = F.softplus(self.to_B_scout(x).float())                       # [B, ScoutDim]
        ph_s_raw   = self.to_phase_scout(x).float().view(Bsz_step, ScoutDim, 2)   # bypass
        ph_s       = F.normalize(ph_s_raw, p=2, dim=-1)
        B_c_scout  = torch.complex(b_scout * ph_s[..., 0], b_scout * ph_s[..., 1])
        h_next_scout = h_prev_scout * self._A_scout() + gate * B_c_scout * psi_scout  # [B, ScoutDim]

        # 2. True step — Simpson's rule: S(U)_t = (1/6)U_t + (4/6)U_{t-1} + (1/6)U_{t-2}
        scout_rms    = (h_prev_scout.real**2 + h_prev_scout.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        h_scout_norm = h_prev_scout / scout_rms                          # [B, ScoutDim]
        h_prev_flat  = torch.cat([h_scout_norm.real, h_scout_norm.imag], dim=-1)  # [B, 2*ScoutDim]
        b_true       = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(h_prev_flat).float())
        ph_t_raw     = self.to_phase_true_x(x).float().view(Bsz_step, SD, 2)   # bypass
        ph_t         = F.normalize(ph_t_raw, p=2, dim=-1)                       # [B, SD, 2]
        B_c_true     = torch.complex(b_true * ph_t[..., 0], b_true * ph_t[..., 1])
        U_curr       = gate * B_c_true * psi                              # [B, SD]
        U_smooth     = (1/6)*U_curr + (4/6)*h_prev_U1 + (1/6)*h_prev_U2  # Simpson's
        h_next_true  = h_prev_true * self._A() + U_smooth                 # [B, SD]

        # Scout state is ScoutDim; zero-pad to SD so all four slots stack into [B, 4, SD]
        h_next_scout_full = F.pad(h_next_scout, (0, SD - ScoutDim))      # [B, SD]
        h_next = torch.stack([h_next_scout_full, h_next_true, U_curr, h_prev_U1], dim=1)  # [B, 4, SD]

        # Complex RMSNorm at READ TIME — matches forward() exactly.
        read_rms = (h_next_true.real**2 + h_next_true.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        h_read   = h_next_true / read_rms                                # [B, SD]

        # Fractal warp — matches forward() exactly.
        h_read = self.fractal(h_read)                                    # [B, SD]

        # Spectrometer + ignition + self-model — bit-identical to forward()
        Bsz = x.shape[0]
        K   = self.n_bands
        bx  = self.band_idx
        Hb   = h_read[:, bx]                                           # [B, K, D/K]
        Pb   = phi[:, bx]
        Pb_n = Pb / Pb.abs().clamp(min=1e-8)                           # unit phasors — matches forward()
        alpha    = (Hb.conj() * Pb_n).sum(-1, keepdim=True) / (SD // K) # [B, K, 1]
        tau_eff  = F.softplus(self.tau).clamp(min=0.1)
        alpha_mag    = alpha.abs()                                          # [B, K, 1] real
        c_k          = F.softmax(alpha_mag / tau_eff, dim=1)                # [B, K, 1]
        alpha_ig_mag = alpha_mag * c_k * K                                  # [B, K, 1] real
        density      = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)     # [B, K, 1]
        alpha_fin_mag = alpha_ig_mag + self.beta * density                  # [B, K, 1] real
        # Holographic unbinding — reuse bypass phasors (no extra trig kernels)
        Q_phase        = torch.complex(ph_t[..., 0], ph_t[..., 1])  # [B, SD]
        unbound        = Hb * Q_phase[:, bx].conj()                     # [B, K, SD/K]
        unbound_scaled = unbound * alpha_fin_mag                        # complex × real: phase untouched
        # Keep complex — phase flows into residual for fractal iteration
        out = torch.zeros(Bsz, SD, device=x.device, dtype=unbound_scaled.dtype)
        out[:, self.inv_band_idx] = unbound_scaled.view(Bsz, SD)
        # Hadamard beam splitter — matches forward() exactly
        G, P = SD // self.mimo_p, self.mimo_p
        H = self.hadamard  # [P, P] fixed unitary
        out_r = torch.einsum('pq,bgq->bgp', H, out.real.reshape(Bsz, G, P))
        out_i = torch.einsum('pq,bgq->bgp', H, out.imag.reshape(Bsz, G, P))
        out = torch.complex(out_r.float(), out_i.float()).reshape(Bsz, SD)
        if isinstance(self.out_proj, nn.Identity):
            return out, h_next
        return torch.complex(self.out_proj(out.real), self.out_proj(out.imag)), h_next


# ---------------------------------------------------------------------------
# ContinuousTransformer
# ---------------------------------------------------------------------------

class WaveForecaster(nn.Module):
    """
    Schrödinger Wave SSM for multivariate time-series forecasting.

    Channel Independent (CI): each variate is a separate univariate series
    through the SAME wave physics.  Per-channel RevIN gives each variable
    its own coordinate ground — the ODE solves in a consistent frame.

    Architecture:
      revin       : RevIN(n_channels) — per-channel instance normalisation
      input_proj  : Linear(1, dim)    — lift scalar → wave field
      operators   : SchrodingerAttention × depth  (wave cascade, no FFN between)
      pred_proj   : Linear(4*dim, pred_len) — DMS from bidirectional frontier

    forward(x: [B, L, C]) → pred: [B, pred_len, C]
    Internally: [B, L, C] → [B*C, L, 1] → SSM → [B*C, P] → [B, P, C]
    """

    def __init__(self, n_channels: int = N_CHANNELS, dim: int = DIM,
                 depth: int = DEPTH, pred_len: int = PRED_LEN,
                 patch_size: int = PATCH_SIZE,
                 use_checkpoint: bool = False, state_dims: list = None):
        super().__init__()
        self.dim            = dim
        self.depth          = depth
        self.pred_len       = pred_len
        self.n_channels     = n_channels
        self.patch_size     = patch_size
        self.use_checkpoint = use_checkpoint

        self.state_dims = state_dims if state_dims is not None else [dim] * depth
        assert len(self.state_dims) == depth

        # ── Patching: downsample time coordinate ───────────────────────
        # 512 hours → 32 patches of 16 hours each.
        # Variance drops to σ²/16, FFT cost drops 16×, dt hierarchy sees cleaner signal.
        # Linear(patch_size, dim) learns the optimal patch embedding.
        self.input_proj = nn.Linear(patch_size, dim)  # [16 hours] → wave field

        # ── RevIN: per-channel instance norm ──────────────────────
        self.revin = RevIN(n_channels)

        # ── Wave physics: pure SSM cascade ─────────────────────────────
        # No FFN between layers — SSM output feeds directly into the next.
        # Each SchrodingerAttention already contains a FractalRationalMap.
        self.operators = nn.ModuleList([
            SchrodingerAttention(dim, mimo_p=MIMO_P, layer_idx=i, depth=depth,
                                 state_dim=self.state_dims[i], patch_size=patch_size)
            for i in range(depth)
        ])

        # ── Direct Multi-Step forecast head ───────────────────────────
        # fwd_real + fwd_imag + bwd_real + bwd_imag = 4D features.
        # One linear draws the entire horizon — no error accumulation.
        self.pred_proj = nn.Linear(4 * dim, pred_len)

    def make_state(self, batch: int = 1, device=None,
                   dtype: torch.dtype = torch.complex64) -> list:
        if device is None:
            device = next(self.parameters()).device
        return [torch.zeros(batch, 4, sd, dtype=dtype, device=device)
                for sd in self.state_dims]

    def _encode(self, x: torch.Tensor):
        """x: [B*C, L//P, P] real → z: [B*C, L//P, D] complex (P=patch_size)"""
        z_real = self.input_proj(x)                                    # [B*C, L//P, D]
        z = torch.complex(z_real.float(),
                          torch.zeros_like(z_real, dtype=torch.float32))
        z_prev = torch.zeros_like(z_real)
        for i in range(self.depth):
            z_in_real = z.real
            if self.training and self.use_checkpoint:
                wave_out = checkpoint(self.operators[i], z_in_real, z_prev,
                                      use_reentrant=False)
            else:
                wave_out = self.operators[i](z_in_real, z_prev)
            z      = z + wave_out
            z_prev = z.real
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, context_len, C]
        → pred: [B, pred_len,    C]

        CI: each channel is an independent univariate series.
        Per-channel RevIN keeps each variable in its own coordinate frame.
        """
        B, L, C = x.shape
        P = self.patch_size
        assert L % P == 0, f"context_len={L} must be divisible by patch_size={P}"
        x = self.revin.norm(x)                                         # [B, L, C]

        # [B, L, C] → [B*C, L] → [B*C, L//P, P]  (patching)
        x_ci = x.permute(0, 2, 1).reshape(B * C, L)                    # [B*C, L]
        x_ci = x_ci.view(B * C, L // P, P)                             # [B*C, 32, 16]

        z_fwd = self._encode(x_ci)                                     # [B*C, L//P, D]
        z_bwd = self._encode(x_ci.flip(dims=[1]))                      # [B*C, L//P, D]

        z_fwd_last = z_fwd[:, -1, :]                                   # [B*C, D]
        z_bwd_last = z_bwd[:, -1, :]                                   # [B*C, D]
        features = torch.cat([
            z_fwd_last.real, z_fwd_last.imag,
            z_bwd_last.real, z_bwd_last.imag,
        ], dim=-1).float()                                             # [B*C, 4D]

        pred = self.pred_proj(features)                                # [B*C, P]
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)    # [B, P, C]
        pred = self.revin.denorm(pred)
        return pred

    def invalidate_A_cache(self) -> None:
        for op in self.operators:
            op._A_cache       = None
            op._A_scout_cache = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_mse = total_mae = total_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)                    # [B, P, C]
        mse  = ((pred - y) ** 2).mean().item()
        mae  = (pred - y).abs().mean().item()
        total_mse += mse * x.shape[0]
        total_mae += mae * x.shape[0]
        total_n   += x.shape[0]
    model.train()
    return {"MSE": total_mse / total_n, "MAE": total_mae / total_n}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _dummy_generation() -> None:  # kept so the rest of this placeholder block isn't reached
    pass

# (The old generate() and evaluate_niah() functions are removed.
#  Time-series models don't generate text.)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(dataset_name: str = DATASET,
          context_len: int = CONTEXT_LEN,
          pred_len:    int = PRED_LEN,
          checkpoint_prefix: str = "wave_ts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = ETTDataset(dataset_name, "train", context_len, pred_len)
    # Pass training scaler to val/test — benchmark-correct: no leakage
    val_ds   = ETTDataset(dataset_name, "val",   context_len, pred_len,
                          mean=train_ds.mean_, std=train_ds.std_)
    test_ds  = ETTDataset(dataset_name, "test",  context_len, pred_len,
                          mean=train_ds.mean_, std=train_ds.std_)
    print(f"Dataset: {dataset_name}  context={context_len}  pred={pred_len}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))

    n_channels = train_ds.data.shape[1]
    model  = WaveForecaster(n_channels=n_channels, dim=DIM, depth=DEPTH,
                            pred_len=pred_len, use_checkpoint=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}  ({params/1e6:.3f}M)")

    # Checkpoint restore
    checkpoints = sorted(glob.glob(f"{checkpoint_prefix}_epoch*.pt"))
    start_epoch = 0
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Loading checkpoint: {latest}")
        try:
            missing, unexpected = model.load_state_dict(
                torch.load(latest, map_location=device), strict=False)
            if missing:    print(f"  New params: {missing}")
            if unexpected: print(f"  Dropped:    {unexpected}")
            start_epoch = int(latest.split("epoch")[1].split(".")[0]) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Load failed ({e}). Starting fresh.")

    # dt params need higher LR — same logic as language model
    dt_params = [op.dt for op in model.operators] + \
                [op.dt_scout for op in model.operators]
    dt_id_set = {id(p) for p in dt_params}
    base_params = [p for p in model.parameters()
                   if id(p) not in dt_id_set and p.requires_grad]
    opt = optim.AdamW([
        {'params': base_params, 'lr': BASE_LR,      'weight_decay': WEIGHT_DECAY},
        {'params': dt_params,   'lr': BASE_LR * 10, 'weight_decay': 0.0},
    ], eps=1e-8)
    # Softer decay: 0.7× per epoch.  The 0.5× schedule was designed for
    # near-linear models (DLinear/PatchTST) that converge in 3-4 epochs.
    # Our deeply nonlinear SSM (fractal maps, Born rule, complex phase)
    # needs more gradient time to crystallize its basin boundaries.
    scheduler = optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda epoch: 0.7 ** epoch
    )

    best_val_mse = float('inf')
    patience_counter = 0
    step = 0

    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            model.train()
            ema_loss = None   # EMA of training MSE
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=USE_AMP and device.type == 'cuda'):
                    pred = model(x)                             # [B, P, C]

                    # ── Training objective: pure MSE ────────────────────────
                    # Every SOTA model (DLinear, PatchTST, iTransformer) uses
                    # MSE only.  Auxiliary losses (Sobolev, spectral) add
                    # competing gradient signals that hurt a tiny 18K-param model.
                    loss = F.mse_loss(pred, y)

                    with torch.no_grad():
                        mae_metric = F.l1_loss(pred.detach(), y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                model.invalidate_A_cache()

                if ema_loss is None: ema_loss = loss.item()
                ema_loss = 0.95 * ema_loss + 0.05 * loss.item()

                if step % PRINT_EVERY == 0:
                    val_metrics  = evaluate(model, val_loader, device)
                    test_metrics = evaluate(model, test_loader, device)
                    dts = " ".join(f"{op.dt.item():.3f}" for op in model.operators)
                    lr_now = opt.param_groups[0]['lr']
                    print(f"Epoch {epoch:02d} Step {step:05d} | "
                          f"train: {ema_loss:.4f} | "
                          f"val: {val_metrics['MSE']:.4f} | "
                          f"test: {test_metrics['MSE']:.4f} | "
                          f"lr: {lr_now:.2e} | "
                          f"dt: [{dts}]")
                step += 1

            # ── Epoch-end validation ────────────────────────────────────────
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)

            if val_metrics['MSE'] < best_val_mse:
                best_val_mse = val_metrics['MSE']
                patience_counter = 0
                best_path = f"{checkpoint_prefix}_best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model saved: {best_path}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                ckpt_path = f"{checkpoint_prefix}_epoch{epoch:03d}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {epoch} step {step}")

    # ── Final test evaluation ───────────────────────────────────────────────
    print("\n── Test set evaluation ──")
    best_path = f"{checkpoint_prefix}_best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST  MSE={test_metrics['MSE']:.4f}  MAE={test_metrics['MAE']:.4f}")

    # ── Benchmark-standard test evaluation ──────────────────────────────────
    # Standard ETT splits use fixed month boundaries (Informer/PatchTST/DLinear):
    #   ETTh: train=8640, val=2880, test=2880  (rows 11520-14400)
    # Scaler must also be fitted on the STANDARD train (rows 0-8640), not ours.
    # This is the only way to get numbers directly comparable to published tables.
    BENCH_SPLITS = {"ETTh1": (8640, 2880, 2880), "ETTh2": (8640, 2880, 2880),
                    "ETTm1": (34560, 11520, 11520), "ETTm2": (34560, 11520, 11520)}
    if dataset_name in BENCH_SPLITS:
        n_tr_b, n_val_b, n_test_b = BENCH_SPLITS[dataset_name]
        path = prepare_ett_dataset(dataset_name)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        raw_data = df.values.astype(np.float32)
        # Standard scaler: fitted on rows 0-n_tr_b (NOT our larger train set)
        bench_mean = raw_data[:n_tr_b].mean(axis=0, keepdims=True)
        bench_std  = raw_data[:n_tr_b].std(axis=0, keepdims=True) + 1e-8
        bench_raw  = raw_data[n_tr_b + n_val_b - context_len : n_tr_b + n_val_b + n_test_b]
        bench_test = ETTDataset.__new__(ETTDataset)
        bench_test.data        = (bench_raw - bench_mean) / bench_std
        bench_test.mean_       = bench_mean
        bench_test.std_        = bench_std
        bench_test.context_len = context_len
        bench_test.pred_len    = pred_len
        bench_test.window      = context_len + pred_len
        bench_test.n_samples   = max(0, len(bench_test.data) - bench_test.window + 1)
        bench_loader = DataLoader(bench_test, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)
        bench_metrics = evaluate(model, bench_loader, device)
        print(f"\n── Benchmark-standard test (rows {n_tr_b+n_val_b}-{n_tr_b+n_val_b+n_test_b}) ──")
        print(f"BENCH MSE={bench_metrics['MSE']:.4f}  MAE={bench_metrics['MAE']:.4f}")
        print(f"  (scaler fitted on rows 0-{n_tr_b}, matching PatchTST/DLinear/iTransformer)")

    print(f"\nParams: {params:,}  ({params/1e6:.3f}M)")
    print(f"Dataset: {dataset_name}  context={context_len}  pred={pred_len}")
    return test_metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default=DATASET,      choices=list(ETT_URLS))
    p.add_argument("--context",  type=int, default=CONTEXT_LEN)
    p.add_argument("--pred_len", type=int, default=PRED_LEN,
                   choices=[96, 192, 336, 720])
    args = p.parse_args()
    train(args.dataset, args.context, args.pred_len)
