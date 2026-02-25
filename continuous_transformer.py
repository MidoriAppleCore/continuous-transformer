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
# Hyper-parameters
# ---------------------------------------------------------------------------
DIM            = int(1024*1.25)
DEPTH          = 8
VOCAB_SIZE     = 256
MIMO_P         = 8     # MIMO group size — channels mix in groups of P (P×P matmul)

BASE_LR        = 3e-4
WEIGHT_DECAY   = 0.01
BATCH_SIZE     = 2
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

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

        # ── Token → wave projections ──────────────────────────────────────
        self.to_psi = nn.Linear(dim, SD * 2)    # excitation wave  ψ (content → wave field)
        nn.init.orthogonal_(self.to_psi.weight)  # orthogonal write basis: key tokens span separate subspaces
        self.to_phi = nn.Linear(dim, SD * 2)    # measurement wave φ (read  → wave field)
        nn.init.orthogonal_(self.to_phi.weight)  # complete orthonormal basis at t=0

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
        b0 = hippo_b_vector(SD)                           # [SD] target at zero input
        b0_inv = torch.log(torch.exp(b0.clamp(min=1e-6)) - 1.0)  # softplus⁻¹
        self.to_B_scout = nn.Linear(dim, SD)              # Scout: input-only coupling
        nn.init.zeros_(self.to_B_scout.weight)
        self.to_B_scout.bias.data.copy_(b0_inv)
        self.to_B_true_x = nn.Linear(dim, SD)             # True: input path (no concat)
        nn.init.zeros_(self.to_B_true_x.weight)
        self.to_B_true_x.bias.data.copy_(b0_inv)          # softplus(b0_inv)=hippo_b at init
        self.to_B_true_h = nn.Linear(SD * 2, SD)          # True: scout state path
        nn.init.zeros_(self.to_B_true_h.weight)
        nn.init.zeros_(self.to_B_true_h.bias)             # zero init — adds nothing at start
        # Phase-selective writing — the "Schrödinger" upgrade.
        # B_t = amp · e^{i·θ(x_t)} instead of amp + 0·i.
        #
        # ── Phase projections (warm-start: uniform random, trainable) ───────────
        # Initialise to_phase_scout and to_phase_true_x with scaled uniform weights.
        # Weight scale 1/√dim: keeps pre-tanh std ≈ 1 given N(0,1) token embeddings,
        # so diversity is present from step 0 without saturating tanh.
        # These are trainable — co-adaptation with to_psi/to_phi is required.
        # Freezing them (previous attempt) collapsed L1 dt→0 because the optimizer
        # found it easier to bypass the wave than to match frozen random phases.
        _phase_w_scale = 1.0 / math.sqrt(dim)
        self.to_phase_scout  = nn.Linear(dim, SD)          # input → write phase (scout)
        nn.init.uniform_(self.to_phase_scout.weight, -_phase_w_scale, _phase_w_scale)
        nn.init.zeros_(self.to_phase_scout.bias)           # bias=0: diversity from W×token only
        self.to_phase_true_x = nn.Linear(dim, SD)          # input → write phase (true)
        nn.init.uniform_(self.to_phase_true_x.weight, -_phase_w_scale, _phase_w_scale)
        nn.init.zeros_(self.to_phase_true_x.bias)          # bias=0: diversity from W×token only
        self.to_phase_true_h = nn.Linear(SD * 2, SD)       # scout state → write phase (true)
        nn.init.zeros_(self.to_phase_true_h.weight)
        nn.init.zeros_(self.to_phase_true_h.bias)
        self.write_gate = nn.Linear(dim, 1)   # scalar gate — WHEN to write
        self.surprise_gain = nn.Parameter(torch.zeros(1))  # predictive coding depth
        self.tau  = nn.Parameter(torch.ones(1))   # ignition temperature (init=1 → soft)
        self.beta = nn.Parameter(torch.zeros(1))  # self-model density weight (init=0 → off)
        # tau=1: standard softmax competition.  Sharpens as bands specialise.
        # beta=0: no self-model at t=0. Activates when density carries gradient signal.
        # ── MIMO grouped channel mixing ───────────────────────────────────────
        # Groups D channels into G = D/P blocks of P; applies a learnable P×P matmul
        # within each block after Born-rule readout.
        # Raises arithmetic intensity ~0.25 → ~2P FLOP/byte, moving the 3050 Ti
        # from memory-bound to compute-bound. Init = identity + ε so t=0 is unchanged.
        assert SD % mimo_p == 0, f"state_dim={SD} must be divisible by mimo_p={mimo_p}"
        self.mimo_p = mimo_p
        G_mimo = SD // mimo_p
        self.mimo_w = nn.Parameter(
            torch.eye(mimo_p).unsqueeze(0).expand(G_mimo, -1, -1).clone()
            + 0.02 * torch.randn(G_mimo, mimo_p, mimo_p)
        )
        # Output projection: maps wave field [SD] → model dim [dim].
        # nn.Identity when SD == dim (default — zero extra params).
        self.out_proj = nn.Linear(SD, dim, bias=False) if SD != dim else nn.Identity()
        self._A_cache = None   # invalidated by ContinuousTransformer.invalidate_A_cache()

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
        B, L, D = x.shape
        SD = self.state_dim       # wave field width — may differ from model dim D

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
        # ── Dual Wave: shared kernel ──────────────────────────────────────
        dt        = F.softplus(self.dt)   # consistent with _A() — always positive
        gamma_eff = F.softplus(self.log_gamma) * (~self.unitary_mask).float()  # zero for unitary dims
        lam       = torch.complex(-gamma_eff, self.omega)  # [SD] unitary dims: pure rotation
        t_axis    = torch.arange(L, device=x.device, dtype=torch.float32)
        kernel_raw = torch.exp(lam.unsqueeze(0) * t_axis.unsqueeze(1) * dt) # [L, SD]

        # Fold Simpson's Rule into the kernel — exploits conv associativity:
        #   (U * Simpson_filter) * K  ≡  U * (Simpson_filter * K)
        # kernel_raw is [L, SD] (no batch dim), so the 3-tap blend costs B× less
        # VRAM than blending the full [B, L, SD] U tensor.  Mathematically identical.
        k_zeros        = torch.zeros(2, SD, dtype=kernel_raw.dtype, device=x.device)
        k_pad          = torch.cat([k_zeros, kernel_raw], dim=0)          # [L+2, SD]
        kernel_simpson = (1/6)*k_pad[2:] + (4/6)*k_pad[1:-1] + (1/6)*k_pad[:-2]  # [L, SD]

        n_fft        = 1 << (2 * L - 1).bit_length()   # next power-of-2 ≥ 2L — cuFFT fast path
        K_freq_scout = torch.fft.fft(kernel_raw.T,     n=n_fft, dim=1)   # [SD, n_fft] Euler
        K_freq_true  = torch.fft.fft(kernel_simpson.T, n=n_fft, dim=1)   # [SD, n_fft] Simpson's

        # ── 1. Scout Wave (input-only LTI — runs first) ───────────────────
        b_scout     = F.softplus(self.to_B_scout(x).float())               # [B, L, D]
        phase_scout = math.pi * torch.tanh(self.to_phase_scout(x).float()) # [B, L, D] ∈(-π,π)
        B_c_scout   = torch.complex(b_scout * torch.cos(phase_scout),
                                    b_scout * torch.sin(phase_scout))      # amp·e^{iθ}
        U_scout   = gate * B_c_scout * psi                                # [B, L, D]
        U_freq_scout = torch.fft.fft(U_scout.permute(0, 2, 1), n=n_fft, dim=2)
        H_scout   = torch.fft.ifft(
            U_freq_scout * K_freq_scout.unsqueeze(0), n=n_fft, dim=2
        )[:, :, :L].permute(0, 2, 1)                                     # [B, L, D]

        # ── 2. Smart Bouncer — causal state-dependent B for the true wave ─
        # H_scout_prev[t] = H_scout[t-1]: what the scout knew *before* step t.
        H_scout_prev = F.pad(H_scout, (0, 0, 1, 0))[:, :-1, :]          # [B, L, D]
        # Normalise before feeding to B-coupling: prevents cascade amplification.
        # Complex RMSNorm: scale by overall magnitude so the phase angle is preserved exactly.
        # LayerNorm(Re) / LayerNorm(Im) independently stretch the axes, converting the
        # unit circle into a random ellipse each step — that would corrupt all stored phases.
        scout_rms      = (H_scout_prev.real**2 + H_scout_prev.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        H_scout_prev_n = H_scout_prev / scout_rms                        # [B, L, SD] unit-phasor-scaled
        H_prev_flat    = torch.cat([H_scout_prev_n.real, H_scout_prev_n.imag], dim=-1)  # [B, L, 2*SD]
        b_true     = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(H_prev_flat).float())
        # Phase must be STRICTLY input-dependent so the write key is identical at
        # query time regardless of what the scout wave has seen since the KV pair.
        # State-dependent phase would encrypt each write with a context password that
        # changes over time — making retrieval impossible at long range.
        # State still controls AMPLITUDE (b_true) for Mamba-style selectivity.
        phase_true = math.pi * torch.tanh(self.to_phase_true_x(x).float())  # [B, L, SD]
        B_c_true   = torch.complex(b_true * torch.cos(phase_true),
                                    b_true * torch.sin(phase_true))        # amp·e^{iθ}

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
        # tau=1 at init → soft competition. Sharpens as bands specialise.
        # Phase preserved — only magnitude |alpha| determines the winner.
        # *K restores total energy (softmax sums to 1, not K).
        tau_eff  = F.softplus(self.tau).clamp(min=0.1)                          # always positive, min 0.1 → max 10× amplification
        c_k      = F.softmax(alpha.abs() / tau_eff, dim=2)                      # [B, L, K, 1]
        alpha_ig = alpha * c_k * K                                              # [B, L, K, 1]

        # Self-model: per-band Born-rule density |h_k|² / (D/K).
        # Each band reports its own self-confidence, not the global field.
        # beta=0 at init → pure external measurement. Activates when density
        # carries useful gradient signal (i.e. confident bands speak louder).
        density   = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)          # [B, L, K, 1]
        alpha_fin = alpha_ig + self.beta * density                             # [B, L, K, 1]

        # Holographic unbinding: demodulate with the EXACT phase used to write.
        # to_phi is still used above for alpha ("how much of key k is in H?").
        # Using Q_phase here makes write-key == read-key by construction; the
        # optimizer no longer needs to discover the conjugate of to_phase_true_x.
        Q_phase        = torch.complex(torch.cos(phase_true), torch.sin(phase_true))  # [B, L, SD]
        unbound        = Hb * Q_phase[:, :, bx].conj()                  # [B, L, K, SD/K]
        unbound_scaled = unbound * alpha_fin.abs()                       # [B, L, K, SD/K]
        out_bands      = unbound_scaled.real + unbound_scaled.imag       # [B, L, K, SD/K]
        out = torch.zeros(B, L, SD, device=x.device)
        out[:, :, bx] = out_bands
        # ── MIMO: grouped cross-channel mixing ───────────────────────────────
        # [B, L, SD] → [B, L, G, P] → P×P matmul per group → [B, L, SD] → out_proj → [B, L, D]
        # Cross-channel coupling within each group; ~2P× more FLOPs than SSM alone.
        G, P = SD // self.mimo_p, self.mimo_p
        out = torch.einsum('gpq,blgq->blgp', self.mimo_w, out.reshape(B, L, G, P)).reshape(B, L, SD)
        return self.out_proj(out)                                               # [B, L, D]

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
        D  = self.dim
        SD = self.state_dim       # wave field width

        psi_raw = self.to_psi(x).float()
        psi     = torch.complex(psi_raw[..., :SD], psi_raw[..., SD:])  # [B, SD]
        phi_raw = self.to_phi(x).float()
        phi     = torch.complex(phi_raw[..., :SD], phi_raw[..., SD:])  # [B, SD]

        gate = torch.sigmoid(self.write_gate(x).float())                # [B, 1]
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)

        # Unpack: h_prev is [B, 4, D] — scout, true, U_{t-1}, U_{t-2}
        h_prev_scout = h_prev[:, 0, :]   # [B, D] complex
        h_prev_true  = h_prev[:, 1, :]   # [B, D] complex
        h_prev_U1    = h_prev[:, 2, :]   # [B, D] U_{t-1}
        h_prev_U2    = h_prev[:, 3, :]   # [B, D] U_{t-2}

        # 1. Scout step (input-only — matches forward's scout wave)
        b_scout      = F.softplus(self.to_B_scout(x).float())
        phase_scout  = math.pi * torch.tanh(self.to_phase_scout(x).float())
        B_c_scout    = torch.complex(b_scout * torch.cos(phase_scout),
                                     b_scout * torch.sin(phase_scout))     # amp·e^{iθ}
        h_next_scout = h_prev_scout * self._A() + gate * B_c_scout * psi  # [B, D]

        # 2. True step — Simpson's rule: S(U)_t = (1/6)U_t + (4/6)U_{t-1} + (1/6)U_{t-2}
        # Complex RMSNorm before B-coupling — matches forward() exactly.
        scout_rms    = (h_prev_scout.real**2 + h_prev_scout.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        h_scout_norm = h_prev_scout / scout_rms                          # [B, SD]
        h_prev_flat  = torch.cat([h_scout_norm.real, h_scout_norm.imag], dim=-1)  # [B, 2*SD]
        b_true       = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(h_prev_flat).float())
        # Input-only phase — matches forward() exactly (no state-dependent password).
        phase_true   = math.pi * torch.tanh(self.to_phase_true_x(x).float())
        B_c_true     = torch.complex(b_true * torch.cos(phase_true),
                                     b_true * torch.sin(phase_true))       # amp·e^{iθ}
        U_curr       = gate * B_c_true * psi                              # [B, D] raw U
        U_smooth     = (1/6)*U_curr + (4/6)*h_prev_U1 + (1/6)*h_prev_U2  # Simpson's
        h_next_true  = h_prev_true * self._A() + U_smooth                 # [B, D]

        h_next = torch.stack([h_next_scout, h_next_true, U_curr, h_prev_U1], dim=1)  # [B, 4, D]

        # Complex RMSNorm at READ TIME — matches forward() exactly.
        read_rms = (h_next_true.real**2 + h_next_true.imag**2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        h_read   = h_next_true / read_rms                                # [B, SD]

        # Spectrometer + ignition + self-model — bit-identical to forward()
        Bsz = x.shape[0]
        K   = self.n_bands
        bx  = self.band_idx
        Hb   = h_read[:, bx]                                           # [B, K, D/K]
        Pb   = phi[:, bx]
        Pb_n = Pb / Pb.abs().clamp(min=1e-8)                           # unit phasors — matches forward()
        alpha    = (Hb.conj() * Pb_n).sum(-1, keepdim=True) / (SD // K) # [B, K, 1]
        tau_eff  = F.softplus(self.tau).clamp(min=0.1)                      # positive, ≥0.1
        c_k      = F.softmax(alpha.abs() / tau_eff, dim=1)                  # [B, K, 1]
        alpha_ig = alpha * c_k * K
        density  = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)   # [B, K, 1]
        alpha_fin = alpha_ig + self.beta * density
        # Holographic unbinding — symmetric with forward(): demodulate with write phase.
        Q_phase        = torch.complex(torch.cos(phase_true), torch.sin(phase_true))  # [B, SD]
        unbound        = Hb * Q_phase[:, bx].conj()                     # [B, K, SD/K]
        unbound_scaled = unbound * alpha_fin.abs()
        out_bands      = unbound_scaled.real + unbound_scaled.imag
        # Inverse permutation gather — same result as zeros+scatter but no allocation.
        # band_idx partitions all SD channels, so inv_band_idx covers [0..SD-1] exactly.
        out = out_bands.view(Bsz, SD)[:, self.inv_band_idx]            # [B, SD]
        # MIMO grouped mixing — matches forward() exactly
        G, P = SD // self.mimo_p, self.mimo_p
        out = torch.einsum('gpq,bgq->bgp', self.mimo_w, out.reshape(Bsz, G, P)).reshape(Bsz, SD)
        return self.out_proj(out), h_next


# ---------------------------------------------------------------------------
# SwiGLU feed-forward block
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block — the non-linear decoder missing from the wave loop.

    out = x + W_down( SiLU(W_gate(norm(x))) * W_up(norm(x)) )

    Pre-norm (LayerNorm before projection) + residual connection.
    W_down zero-init → pure identity at step 0 → loaded checkpoints unaffected.
    hidden = dim * 8 // 3  (≈2.67×, standard LLaMA/PaLM SwiGLU ratio).
    bias=False throughout — LayerNorm already centres activations.

    Why this matters for MQAR:
      The Born-rule measurement Re(h·φ*) is LINEAR in h.  It can route the
      correct wave amplitude to the output channel, but it cannot *translate*
      a continuous-valued amplitude back into a sharp discrete token identity.
      SwiGLU adds the non-linear lookup table that makes that translation cheap.
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden       = dim * 8 // 3          # ≈2.67× — SwiGLU standard
        self.norm    = nn.LayerNorm(dim)
        self.w_gate  = nn.Linear(dim, hidden, bias=False)
        self.w_up    = nn.Linear(dim, hidden, bias=False)
        self.w_down  = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.w_down.weight)   # identity residual at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D]  →  [..., D]  (works for both [B,L,D] and [B,D])"""
        z = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(z)) * self.w_up(z))


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

        self.embed     = nn.Embedding(vocab, dim)
        # Independent physics per layer: each has its own omega, log_gamma, dt.
        # Layer 0 tends to learn letter-level patterns (fast decay).
        # Layer depth-1 tends to learn theme-level patterns (slow decay).
        self.operators = nn.ModuleList([
            SchrodingerAttention(dim, mimo_p=MIMO_P, layer_idx=i, depth=depth,
                                 state_dim=self.state_dims[i])
            for i in range(depth)
        ])
        self.ffn       = nn.ModuleList([
            torch.compile(SwiGLU(dim), fullgraph=True) for _ in range(depth)
        ])
        self.depth_emb = nn.Parameter(torch.randn(depth, dim) * 0.02)
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
        """x: [B, L]  →  logits [B, L, V], z [B, L, D]"""
        z      = self.embed(x)                                          # [B, L, D]
        z_prev = torch.zeros_like(z)   # layer 0 has no prediction yet
        for i in range(self.depth):
            z_in = z + self.depth_emb[i]
            if self.training and self.use_checkpoint:
                z = checkpoint(self.operators[i], z_in, z_prev, use_reentrant=False)
            else:
                z = self.operators[i](z_in, z_prev)
            z = self.ffn[i](z)                                          # non-linear decode
            z_prev = z   # this layer's output = next layer's prediction
        logits = self.out_proj(z)                                       # [B, L, V]
        return logits, z

    def invalidate_A_cache(self) -> None:
        """Call after optimizer.step(). Forces _A() to recompute from updated params.
        Without this, all forward_step calls within a collect+replay cycle share
        the same cached A tensor — 12,288 exp() calls → 6."""
        for op in self.operators:
            op._A_cache = None

    def forward_step(self, x: torch.Tensor, h_prev: list):
        """
        Single-token recurrent step.

        x      : [B, 1]   token indices
        h_prev : list     from make_state() or prior step —
                          List[Tensor[B, 4, state_dim_i]] one tensor per layer
        returns: logits [B, 1, V],
                 h_new  List[Tensor[B, 4, state_dim_i]]
        """
        z        = self.embed(x).squeeze(1)                            # [B, D]
        z_prev   = torch.zeros_like(z)
        h_layers = []
        for i in range(self.depth):
            z_in = z + self.depth_emb[i]
            z, h_i = self.operators[i].forward_step(z_in, h_prev[i], z_prev=z_prev)
            z = self.ffn[i](z)                                         # non-linear decode
            z_prev = z
            h_layers.append(h_i)
        h_new  = h_layers                                              # list[B, 4, state_dim_i]
        logits = self.out_proj(z).unsqueeze(1)                         # [B, 1, V]
        return logits, h_new


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

    data_path  = prepare_wikipedia_dataset()
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
    dt_params   = [op.dt for op in model.operators]
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
                    generate(model, "<|user|>\nTell me a story about a dog.\n<|assistant|>\n", device=str(device))
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
