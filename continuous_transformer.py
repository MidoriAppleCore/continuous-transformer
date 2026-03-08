"""
Continuous Transformer — Schrödinger Wave Attention
====================================================
Rebuilt from the init commit. SpectralField replaced with SchrodingerAttention.

Physics (per layer, per step):
  Evolution:     h_t  = h_{t-1} · exp((-γ + iω)·dt)
  Born rule:     a_t  = σ(W · Re(ψ_t · h_{t-1}*))   ← interference → attention
  Superposition: h_t  = h_t + a_t · ψ_t
  Manifold lock: h_t  = exact_projection(Re) + i·exact_projection(Im)
  Measurement:   out  = Re(h_t · φ_t*)

Training   O(L log L) — parallel FFT convolution (local-conv attention approx)
Generation O(1)       — true recurrent forward_step, infinite context
"""

# ---------------------------------------------------------------------------
# Hyper-parameters  (Micro-Leviathan — < 1M params)
# ---------------------------------------------------------------------------
DIM            = 64      # Squeezed phase space — fractal basins, not lookup tables
DEPTH          = 4        # 4 Julia-set iterations across layers
VOCAB_SIZE     = 256      # Byte-level
MIMO_P         = 4        # Smaller channel-mixing blocks (128 / 4 = 32 groups)

BASE_LR        = 5e-4     # Higher LR — tiny models need more of a kick
WEIGHT_DECAY   = 0.01
BATCH_SIZE     = 8        # DIM=128 is 4x smaller; batch=2 gives 4x less VRAM than before
# Variable sequence length curriculum.
# Each entry is (seq_len, sampling_probability).
# Most steps are short (fast + diverse data); rare long steps train memory.
# MAX_SEQ_LENGTH is derived automatically as the largest entry.
SEQ_SCHEDULE = [
    (256,  0.40),   # 40% — fast, maximum data diversity
    (512,  0.30),   # 30% — sentence / short paragraph
    (1024, 0.20),   # 20% — multi-paragraph context
    (2048, 0.10),   # 10% — long context, exercises wave memory
]
MAX_SEQ_LENGTH = max(l for l, _ in SEQ_SCHEDULE)
GRAD_CLIP      = 1.0

CHECKPOINT_EVERY = 1000
PRINT_EVERY      = 100

USE_AMP      = True   # bfloat16 autocast — ~2x on Ampere, ~4x on A100 tensor cores
NUM_WORKERS  = 4      # DataLoader prefetch workers (set 0 to debug)

GEN_LENGTH      = 300
GEN_TEMPERATURE = 0.85
GEN_TOP_P       = 0.9

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import math, os, glob, mmap, urllib.request
import numpy as np
import torch
from scipy.fft import next_fast_len
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------
# Dataset  (verbatim from init commit)
# ---------------------------------------------------------------------------

def prepare_tinystories_dataset(cache_dir: str = "~/.cache/continuous_transformer") -> str:
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "tinystories_train.txt")
    if not os.path.exists(train_path):
        print("Downloading TinyStories training data...")
        url = (
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
            "TinyStoriesV2-GPT4-train.txt"
        )
        urllib.request.urlretrieve(url, train_path)
        print(f"Downloaded to {train_path}")
    file_size = os.path.getsize(train_path)
    print(f"Using dataset: {train_path} ({file_size/1e6:.1f} MB)")
    return train_path


class TinyStoriesDataset(Dataset):
    def __init__(self, data_path: str, seq_length: int, stride: int = 512):
        self.data_path  = data_path
        self.seq_length = seq_length
        self.stride     = stride
        self.file_size  = os.path.getsize(data_path)
        self.length     = (self.file_size - seq_length - 1) // stride
        self._file  = None
        self._mmap  = None

    def _ensure_mmap(self):
        if self._mmap is None:
            self._file = open(self.data_path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        mm  = self._ensure_mmap()
        pos = idx * self.stride
        mm.seek(pos)
        chunk = mm.read(self.seq_length + 1)
        try:
            text = chunk.decode("utf-8", errors="ignore")
        except Exception:
            text = chunk.decode("latin-1", errors="ignore")
        text = "".join(c for c in text if ord(c) < 128)
        if len(text) < self.seq_length + 1:
            text = text + " " * (self.seq_length + 1 - len(text))
        x = torch.tensor([ord(c) for c in text[:self.seq_length]],  dtype=torch.long)
        y = torch.tensor([ord(c) for c in text[1:self.seq_length+1]], dtype=torch.long)
        return x, y

    def __del__(self):
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()


def prepare_openhermes_dataset(cache_dir: str = "~/.cache/continuous_transformer") -> str:
    """
    Download OpenHermes-2.5 and flatten to a single UTF-8 text file
    using the same format as TinyStories so TinyStoriesDataset works unchanged.

    Each conversation becomes:
        <|user|>\n{prompt}\n<|assistant|>\n{response}\n<|end|>\n\n
    Multi-turn conversations stack multiple user/assistant pairs before <|end|>.
    All non-ASCII bytes are dropped so VOCAB_SIZE=256 is still valid.
    """
    cache_dir  = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    out_path   = os.path.join(cache_dir, "openhermes_train.txt")
    if os.path.exists(out_path):
        print(f"Using cached dataset: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading OpenHermes-2.5 (this may take a few minutes)...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"Downloaded {len(ds)} conversations. Flattening to text...")

    with open(out_path, "w", encoding="ascii", errors="ignore") as f:
        for row in ds:
            # Each row has a 'conversations' list of {from, value} dicts
            convs = row.get("conversations", [])
            if not convs:
                continue
            buf = []
            for turn in convs:
                role  = turn.get("from", "").lower()   # 'human' or 'gpt'
                value = turn.get("value", "").strip()
                if not value:
                    continue
                tag = "<|user|>" if role == "human" else "<|assistant|>"
                buf.append(f"{tag}\n{value}\n")
            if buf:
                f.write("".join(buf) + "<|end|>\n\n")

    size = os.path.getsize(out_path)
    print(f"Saved: {out_path} ({size/1e6:.1f} MB)")
    return out_path


def prepare_wikipedia_dataset(cache_dir: str = "~/.cache/continuous_transformer") -> str:
    """
    Download English Wikipedia (20220301.en) and flatten to a single ASCII text file.
    Each article becomes:
        = Title =\n\n{body text}\n\n
    All non-ASCII bytes are dropped so VOCAB_SIZE=256 remains valid.
    ~3.5GB on disk after flattening — richer and more linguistically diverse
    than OpenHermes for pretraining from scratch.
    Pairs of Wikipedia sections give coherent long-form prose: good for
    exercising the wave memory at L=1024/2048 in the seqlen curriculum.
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    out_path  = os.path.join(cache_dir, "wikipedia_train.txt")
    if os.path.exists(out_path):
        print(f"Using cached dataset: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading Wikipedia 20231101.en — this may take 10-20 minutes...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    print(f"Downloaded {len(ds)} articles. Flattening to text...")

    with open(out_path, "w", encoding="ascii", errors="ignore") as f:
        for row in ds:
            title = row.get("title", "").strip()
            text  = row.get("text",  "").strip()
            if not text:
                continue
            # WikiText-style heading so slow wave layers can learn document structure
            f.write(f" = {title} = \n\n{text}\n\n")

    size = os.path.getsize(out_path)
    print(f"Saved: {out_path} ({size/1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Math utilities  (verbatim from init commit)
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
    Fractal Feed-Forward — natively complex, replaces SwiGLU entirely.

    The residual stream z is complex throughout the network. Each layer's
    fractal map is one iteration of the Julia set recurrence z_{ℓ+1} = R(z_ℓ).
    Across 4 layers, the signal passes through 4 fractal iterations, carving
    progressively finer basin boundaries into the complex phase space.

    Pipeline:  z_complex → CRMSNorm (pure unit-sphere) → FractalRationalMap → residual

    The norm projects onto the unit hyper-sphere: |z|=1 everywhere.
    The fractal map is the SOLE arbiter of scale. No fighting parameters.

    Total params: 6 × dim per layer (pure fractal only).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps     = eps
        self.fractal = FractalRationalMap(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., D] complex → [..., D] complex"""
        # Pure unit-sphere projection: |z_normed| = 1 everywhere
        rms = (z.real**2 + z.imag**2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        z_normed = z / rms
        # Julia-set rational map — one iteration of the fractal recurrence
        return z + self.fractal(z_normed)


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
                 layer_idx: int = 0, depth: int = 1, state_dim: int = None):
        super().__init__()
        self.dim    = dim
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
        dt_max, dt_min = 0.1, 0.01
        ratio          = layer_idx / max(depth - 1, 1)
        target_dt      = dt_max * (dt_min / dt_max) ** ratio   # geometric interp
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

class ContinuousTransformer(nn.Module):
    """
    Depth-stacked SchrodingerAttention layers.

    forward()       — parallel training,  O(L log L)
    forward_step()  — recurrent inference, O(1)
    """

    def __init__(self, dim: int = 256, depth: int = 6, vocab: int = 256,
                 use_checkpoint: bool = True, state_dims: list = None):
        super().__init__()
        self.dim            = dim
        self.depth          = depth
        self.use_checkpoint = use_checkpoint
        # state_dims: per-layer wave field widths. Default = [dim]*depth (unchanged behaviour).
        # Increase on deep/slow layers for more long-term memory capacity.
        # Example (depth=4): state_dims=[256, 256, 512, 1024]
        #   L3 → 512 complex slots vs 128 at default — ideal for game agents and long text.
        self.state_dims = state_dims if state_dims is not None else [dim] * depth
        assert len(self.state_dims) == depth, \
            f"len(state_dims)={len(self.state_dims)} must equal depth={depth}"

        # ── Fixed Spectral Basis (0 parameters) ───────────────────────────
        # Each byte is a unique physical boundary condition — a superposition of
        # prime-harmonic oscillators frozen at construction time.
        # The model never learns 'what A is'; A is defined as a rigid harmonic
        # signature.  Intelligence lives entirely in the fractal wave dynamics,
        # not in a lookup table.  Deletes ~vocab×dim params from the model.
        # Prime frequencies guarantee no harmonic overlap across the 256-byte vocab.
        _primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
                   59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131]
        with torch.no_grad():
            freqs  = torch.tensor(_primes[:dim // 2], dtype=torch.float32)
            phases = torch.linspace(0, 2 * math.pi, vocab).unsqueeze(1)  # [V, 1]
            basis  = torch.cat([
                torch.sin(phases * freqs),   # [V, dim//2]
                torch.cos(phases * freqs),   # [V, dim//2]
            ], dim=-1)                        # [V, dim]
        self.register_buffer('fixed_basis', basis)  # frozen, moves with .to(device)
        # Independent physics per layer: each has its own omega, log_gamma, dt.
        # Layer 0 tends to learn letter-level patterns (fast decay).
        # Layer depth-1 tends to learn theme-level patterns (slow decay).
        self.operators = nn.ModuleList([
            SchrodingerAttention(dim, mimo_p=MIMO_P, layer_idx=i, depth=depth,
                                 state_dim=self.state_dims[i])
            for i in range(depth)
        ])
        self.ffn       = nn.ModuleList([
            FractalFFN(dim) for _ in range(depth)
        ])
        self.out_proj  = nn.Linear(dim, vocab)

    def make_state(self, batch: int = 1, device=None,
                   dtype: torch.dtype = torch.complex64) -> list:
        """Zero initial hidden state for all layers.

        Returns List[Tensor[batch, 4, state_dim_i]] — one tensor per layer.
        Use this instead of torch.zeros() so per-layer state_dims are respected.
        Slots per layer:  [0] scout wave  [1] true wave
                          [2] U_{t-1}     [3] U_{t-2}  (Simpson lags)
        """
        if device is None:
            device = next(self.parameters()).device
        return [torch.zeros(batch, 4, sd, dtype=dtype, device=device)
                for sd in self.state_dims]

    def forward(self, x: torch.Tensor):
        """x: [B, L]  →  logits [B, L, V], z [B, L, D]

        Residual stream z is complex throughout — each layer's fractal FFN
        is one iteration of the Julia set recurrence. Phase accumulates
        across layers; the fractal carves progressively finer basins.
        """
        z_real = F.embedding(x, self.fixed_basis)                         # [B, L, D] real — fixed harmonic basis
        z = torch.complex(z_real.float(),
                          torch.zeros_like(z_real, dtype=torch.float32)) # [B, L, D] complex
        z_prev = torch.zeros_like(z_real)   # predictive coding uses real content
        for i in range(self.depth):
            z_in_real = z.real                 # wave carries its own layer identity via phase history
            if self.training and self.use_checkpoint:
                wave_out = checkpoint(self.operators[i], z_in_real, z_prev,
                                      use_reentrant=False)
            else:
                wave_out = self.operators[i](z_in_real, z_prev)
            z = z + wave_out                                            # complex + complex
            z = self.ffn[i](z)                                          # fractal iteration
            z_prev = z.real   # next layer's prediction = real content
        logits = self.out_proj(z.real)                                   # [B, L, V]
        return logits, z.real

    def invalidate_A_cache(self) -> None:
        """Call after optimizer.step(). Forces _A() and _A_scout() to recompute.
        Without this, forward_step calls share stale A tensors from before the step."""
        for op in self.operators:
            op._A_cache       = None
            op._A_scout_cache = None

    def forward_step(self, x: torch.Tensor, h_prev: list):
        """
        Single-token recurrent step — complex residual stream.

        x      : [B, 1]   token indices
        h_prev : list     from make_state() or prior step
        returns: logits [B, 1, V],
                 h_new  List[Tensor[B, 4, state_dim_i]]
        """
        z_real   = F.embedding(x, self.fixed_basis).squeeze(1)          # [B, D] real — fixed harmonic basis
        z        = torch.complex(z_real.float(),
                                 torch.zeros_like(z_real, dtype=torch.float32))
        z_prev   = torch.zeros_like(z_real)
        h_layers = []
        for i in range(self.depth):
            z_in_real = z.real                 # wave carries its own layer identity via phase history
            wave_out, h_i = self.operators[i].forward_step(
                z_in_real, h_prev[i], z_prev=z_prev
            )
            z = z + wave_out                                           # complex + complex
            z = self.ffn[i](z)                                         # fractal iteration
            z_prev = z.real
            h_layers.append(h_i)
        logits = self.out_proj(z.real).unsqueeze(1)                     # [B, 1, V]
        return logits, h_layers


# ---------------------------------------------------------------------------
# Generation — true O(1) infinite context via forward_step
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: nn.Module, prompt: str, device: str = "cuda",
             length: int = GEN_LENGTH, temperature: float = GEN_TEMPERATURE,
             top_p: float = GEN_TOP_P) -> None:
    model.eval()
    tokens = [ord(c) for c in prompt]
    print(f"\nGeneration (O(1) recurrent): {prompt}", end="", flush=True)

    # Initialize empty wave fields: scout, true, U_{t-1}, U_{t-2} per layer
    h = model.make_state(1, device)

    # Encode the prompt by stepping through it token by token
    for tok in tokens:
        t = torch.tensor([[tok]], device=device)
        _, h = model.forward_step(t, h)

    # Autoregressive generation — infinite context, O(1) per step
    for _ in range(length):
        t = torch.tensor([[tokens[-1]]], device=device)
        logits, h = model.forward_step(t, h)

        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cumsum > top_p).float()
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0]  = 0
        probs[idx[cutoff.bool()]] = 0
        probs = probs / probs.sum()

        next_tok = torch.multinomial(probs, 1).item()
        tokens.append(next_tok)
        print(chr(next_tok), end="", flush=True)

    print("\n")
    model.train()


# ---------------------------------------------------------------------------
# Needle-in-a-Haystack evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_niah(model: nn.Module, device: str = "cuda",
                  n_trials: int = 20,
                  haystack_len: int = 2000,
                  needle_key_len: int = 8,
                  needle_val_len: int = 8,
                  depths: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    """
    Synthetic Needle-in-a-Haystack test (byte-level, VOCAB=256 compatible).

    For each trial and each needle depth:
      1. Build a haystack of `haystack_len` random printable ASCII bytes.
      2. Insert "KEY=<key> VALUE=<val>" at position floor(depth * haystack_len).
      3. Append "Q: VALUE after KEY=<key>? A: " as the query.
      4. Step through the full context with forward_step (O(1) per token).
      5. Greedily decode `needle_val_len` tokens and check exact match.

    Returns dict: {depth: accuracy} + 'mean' key.
    """
    import random, string
    model.eval()

    printable = [c for c in string.printable if c.isprintable() and ord(c) < 128
                 and c not in '<>=\n\r']

    depth_hits   = {d: 0 for d in depths}
    depth_trials = {d: 0 for d in depths}

    for trial in range(n_trials):
        # Fresh random key and value for each trial
        key = ''.join(random.choices(string.ascii_uppercase, k=needle_key_len))
        val = ''.join(random.choices(string.digits, k=needle_val_len))
        needle = f"KEY={key} VALUE={val} "

        haystack_chars = random.choices(printable, k=haystack_len)

        for depth in depths:
            insert_pos = int(depth * haystack_len)
            chars = haystack_chars[:insert_pos] + list(needle) + haystack_chars[insert_pos:]
            query = f"Q: VALUE after KEY={key}? A: "
            full  = "".join(chars) + query

            # Encode as bytes and step through
            tokens = [ord(c) for c in full if ord(c) < 128]
            h = model.make_state(1, device)
            for tok in tokens:
                t = torch.tensor([[tok]], device=device)
                _, h = model.forward_step(t, h)

            # Greedy decode val_len tokens
            decoded = []
            tok = tokens[-1]
            for _ in range(needle_val_len):
                t = torch.tensor([[tok]], device=device)
                logits, h = model.forward_step(t, h)
                tok = logits[0, -1].argmax().item()
                decoded.append(chr(tok) if 32 <= tok < 128 else '?')

            predicted = "".join(decoded)
            depth_hits[depth]   += int(predicted == val)
            depth_trials[depth] += 1

    results = {d: depth_hits[d] / depth_trials[d] for d in depths}
    results['mean'] = sum(results[d] for d in depths) / len(depths)

    print(f"\n── NIAH (haystack={haystack_len}, key={needle_key_len}B, val={needle_val_len}B) ──")
    for d in depths:
        bar = "█" * int(results[d] * 20) + "░" * (20 - int(results[d] * 20))
        print(f"  depth {d:.2f}: {bar}  {results[d]*100:.1f}%")
    print(f"  mean accuracy: {results['mean']*100:.1f}%\n")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(checkpoint_prefix: str = "continuous_transformer"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_path  = prepare_tinystories_dataset()
    # Dataset always loads MAX_SEQ_LENGTH; variable_collate_fn truncates each
    # batch to a length sampled from SEQ_SCHEDULE before it hits the model.
    dataset    = TinyStoriesDataset(data_path, seq_length=MAX_SEQ_LENGTH, stride=256)

    _lengths = [l for l, _ in SEQ_SCHEDULE]
    _probs   = np.array([p for _, p in SEQ_SCHEDULE], dtype=np.float64)
    _probs  /= _probs.sum()   # normalise in case they don't sum to 1

    def variable_collate_fn(batch):
        """Sample one seq_len from SEQ_SCHEDULE and truncate the whole batch."""
        L   = int(np.random.choice(_lengths, p=_probs))
        xs  = torch.stack([item[0][:L] for item in batch])
        ys  = torch.stack([item[1][:L] for item in batch])
        return xs, ys

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=variable_collate_fn,
    )

    model  = ContinuousTransformer(dim=DIM, depth=DEPTH, vocab=VOCAB_SIZE).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params/1e6:.2f}M")

    # Checkpoint restore
    checkpoints = glob.glob(f"{checkpoint_prefix}_step*.pt")
    if checkpoints:
        steps = [int(cp.split("step")[1].split(".")[0]) for cp in checkpoints]
        latest    = max(steps)
        ckpt_path = f"{checkpoint_prefix}_step{latest:05d}.pt"
        print(f"Loading checkpoint: {ckpt_path}")
        try:
            missing, unexpected = model.load_state_dict(
                torch.load(ckpt_path, map_location=device), strict=False
            )
            if missing:
                print(f"  New params (random init): {missing}")
            if unexpected:
                print(f"  Dropped params: {unexpected}")
            start_step = latest
            print(f"Resuming from step {start_step}")
        except Exception as e:
            print(f"Load failed ({e}). Starting fresh.")
            start_step = 0
    else:
        start_step = 0
        print("No checkpoint found. Starting fresh.")

    # torch.compile disabled locally: torchinductor has no complex op codegen,
    # so it adds graph-materialization memory overhead with zero fusion benefit.
    # Re-enable on Colab A100 where complex CUDA kernels are available:
    #   model = torch.compile(model)

    # dt params need 30x higher LR: softplus'(softplus_inv(0.001)) ≈ 0.001, so the
    # gradient reaching dt_raw is 1000x smaller than a normal param. weight_decay=0
    # because L2 on a log-timescale param biases it toward 0 (faster decay = forgetting).
    dt_params   = [op.dt       for op in model.operators] + \
                  [op.dt_scout for op in model.operators]
    dt_id_set   = {id(p) for p in dt_params}
    base_params = [p for p in model.parameters()
                   if id(p) not in dt_id_set and p.requires_grad]
    opt = optim.AdamW([
        {'params': base_params, 'initial_lr': BASE_LR,       'lr': BASE_LR,       'weight_decay': WEIGHT_DECAY},
        {'params': dt_params,   'initial_lr': BASE_LR * 30,  'lr': BASE_LR * 30,  'weight_decay': 0.0},
    ], eps=1e-8, fused=True)
    ema_loss = None
    step     = start_step

    try:
        while True:
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                # bfloat16 matmuls (linear layers) + float32 wave physics / loss
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=USE_AMP and device.type == 'cuda'):
                    logits, _ = model(x)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), y.view(-1))

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # clip_grad_norm_ returns pre-clip total norm — reuse for reactive LR
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                with torch.no_grad():
                    if ema_loss is None:
                        ema_loss = loss.item()
                    ema_loss      = 0.95 * ema_loss + 0.05 * loss.item()
                    stability     = 1.0 / (grad_norm.item() + 1e-6)
                    lr_multiplier = 1.0 / (1.0 + np.exp(-(stability - 0.5)))
                    for pg in opt.param_groups:
                        # Scale each group's own initial_lr — preserves the 30x dt ratio
                        pg["lr"] = pg["initial_lr"] * lr_multiplier

                opt.step()
                # A = exp((-γ+iω)·dt) is cached; must recompute after dt changes.
                # Without this, generate() and any forward_step call in the same
                # process would use stale physics from before the last optimizer step.
                model.invalidate_A_cache()

                if step % PRINT_EVERY == 0:
                    acc = (logits.argmax(-1) == y).float().mean()
                    dts = " ".join(f"{op.dt.item():.3f}" for op in model.operators)
                    current_lr = opt.param_groups[0]["lr"]
                    print(
                        f"Step {step:05d} | L={x.shape[1]:4d} | Loss: {loss.item():.4f} | "
                        f"Acc: {acc.item()*100:.1f}% | LR: {current_lr:.2e} | "
                        f"dt: [{dts}]"
                    )

                if step % CHECKPOINT_EVERY == 0 and step > start_step:
                    generate(model, "The capital of France is ", device=str(device))
                    ckpt_path = f"{checkpoint_prefix}_step{step:05d}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")

                step += 1

    except KeyboardInterrupt:
        print(f"\nInterrupted at step {step}")
        final = f"{checkpoint_prefix}_step{step:05d}.pt"
        torch.save(model.state_dict(), final)
        print(f"Saved: {final}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
