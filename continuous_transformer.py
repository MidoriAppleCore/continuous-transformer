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
DIM            = int(512)
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

    def __init__(self, dim: int, n_bands: int = 4, mimo_p: int = 8):
        super().__init__()
        self.dim    = dim
        self.n_bands = n_bands   # spectral bands: each reads D/K independent modes
        assert dim % n_bands == 0, f"dim={dim} must be divisible by n_bands={n_bands}"

        # ── Wave physics ──────────────────────────────────────────────────
        self.omega     = nn.Parameter(hippo_freqs(dim))        # [D] HiPPO rotation
        self.log_gamma = nn.Parameter(torch.randn(dim) - 3.0)  # [D] dispersive decay
        self.dt        = nn.Parameter(torch.tensor(0.1))       # integration step

        # ── Token → wave projections ──────────────────────────────────────
        self.to_psi = nn.Linear(dim, dim * 2)   # excitation wave  ψ (content)
        self.to_phi = nn.Linear(dim, dim * 2)   # measurement wave φ (read)
        nn.init.orthogonal_(self.to_phi.weight)  # complete orthonormal basis at t=0

        # ── HiPPO-aligned band boundaries ────────────────────────────────
        # hippo_freqs returns exp(-sqrt(2n+1)/max · log10000), monotone ↓ in n.
        # Argsort gives indices ordered slow→fast (low freq → high freq).
        # We cut that ordering into K equal-count bands so each band spans
        # one quartile of the HiPPO frequency spectrum, not a raw index range.
        freqs     = hippo_freqs(dim)                          # [D] monotone ↓
        order     = torch.argsort(freqs, descending=True)     # slow→fast
        band_size = dim // n_bands
        bands     = order.view(n_bands, band_size)            # [K, D/K] indices
        self.register_buffer('band_idx', bands)               # [K, D/K] long

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
        b0 = hippo_b_vector(dim)                          # [D] target at zero input
        b0_inv = torch.log(torch.exp(b0.clamp(min=1e-6)) - 1.0)  # softplus⁻¹
        self.to_B_scout = nn.Linear(dim, dim)             # Scout: input-only coupling
        nn.init.zeros_(self.to_B_scout.weight)
        self.to_B_scout.bias.data.copy_(b0_inv)
        self.to_B_true_x = nn.Linear(dim, dim)            # True: input path (no concat)
        nn.init.zeros_(self.to_B_true_x.weight)
        self.to_B_true_x.bias.data.copy_(b0_inv)          # softplus(b0_inv)=hippo_b at init
        self.to_B_true_h = nn.Linear(dim * 2, dim)        # True: scout state path
        nn.init.zeros_(self.to_B_true_h.weight)
        nn.init.zeros_(self.to_B_true_h.bias)             # zero init — adds nothing at start
        self.write_gate = nn.Linear(dim, 1)   # scalar gate — WHEN to write
        # Simpson's rule causal filter [1/6, 4/6, 1/6] — zero parameters.
        # Applied as depthwise conv1d (training) / 3-tap recurrence (inference).
        # Shape [D, 1, 3] for F.conv1d groups=dim.
        simp = torch.tensor([1/6, 4/6, 1/6], dtype=torch.float32)
        self.register_buffer('simpson_kernel', simp.view(1, 1, 3).expand(dim, -1, -1).clone())
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
        assert dim % mimo_p == 0, f"dim={dim} must be divisible by mimo_p={mimo_p}"
        self.mimo_p = mimo_p
        G_mimo = dim // mimo_p
        self.mimo_w = nn.Parameter(
            torch.eye(mimo_p).unsqueeze(0).expand(G_mimo, -1, -1).clone()
            + 0.02 * torch.randn(G_mimo, mimo_p, mimo_p)
        )

    # ------------------------------------------------------------------
    def _A(self) -> torch.Tensor:
        """Constant evolution factor A = exp((-γ + iω)·|dt|)  [D]  complex"""
        dt = self.dt.abs()
        return torch.exp(torch.complex(
            -torch.exp(self.log_gamma) * dt,
            self.omega * dt,
        ))

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

        psi_raw = self.to_psi(x).float()                                # fp32 — complex64 needs float32 parts
        psi     = torch.complex(psi_raw[..., :D], psi_raw[..., D:])    # [B, L, D]
        phi_raw = self.to_phi(x).float()
        phi     = torch.complex(phi_raw[..., :D], phi_raw[..., D:])    # [B, L, D]

        # Predictive coding: gate scales with surprise (deviation from prediction).
        # When z_prev perfectly predicts x, surprise→0 and gate closes.
        # Wave only updates when it encounters something unexpected.
        gate = torch.sigmoid(self.write_gate(x).float())                # [B, L, 1] real
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, L, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)
        # ── Dual Wave: shared kernel ──────────────────────────────────────
        dt     = self.dt.abs()
        lam    = torch.complex(-torch.exp(self.log_gamma), self.omega)  # [D]
        t_axis = torch.arange(L, device=x.device, dtype=torch.float32)
        kernel = torch.exp(lam.unsqueeze(0) * t_axis.unsqueeze(1) * dt) # [L, D]
        n_fft  = 2 * L
        K_freq = torch.fft.fft(kernel.T, n=n_fft, dim=1)               # [D, n_fft]

        # ── 1. Scout Wave (input-only LTI — runs first) ───────────────────
        b_scout   = F.softplus(self.to_B_scout(x).float())               # [B, L, D]
        B_c_scout = torch.complex(b_scout, torch.zeros_like(b_scout))
        U_scout   = gate * B_c_scout * psi                                # [B, L, D]
        U_freq_scout = torch.fft.fft(U_scout.permute(0, 2, 1), n=n_fft, dim=2)
        H_scout   = torch.fft.ifft(
            U_freq_scout * K_freq.unsqueeze(0), n=n_fft, dim=2
        )[:, :, :L].permute(0, 2, 1)                                     # [B, L, D]

        # ── 2. Smart Bouncer — causal state-dependent B for the true wave ─
        # H_scout_prev[t] = H_scout[t-1]: what the scout knew *before* step t.
        H_scout_prev = F.pad(H_scout, (0, 0, 1, 0))[:, :-1, :]          # [B, L, D]
        H_prev_flat  = torch.cat([H_scout_prev.real, H_scout_prev.imag], dim=-1)  # [B, L, 2D]
        b_true   = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(H_prev_flat).float())  # [B, L, D]
        B_c_true = torch.complex(b_true, torch.zeros_like(b_true))

        # ── 3. True Wave — Simpson's rule quadrature (4th-order) ────────────
        # S(U)_t = (1/6)·U_t + (4/6)·U_{t-1} + (1/6)·U_{t-2}
        # Applied as depthwise causal conv1d with fixed kernel [1/6, 4/6, 1/6].
        # Real and imaginary parts handled separately (linear op → equivalent).
        # padding=2 then slice [:-2] gives exact causal alignment.
        U_true_raw = gate * B_c_true * psi                                # [B, L, D] complex
        Ur = U_true_raw.real.float().permute(0, 2, 1)                    # [B, D, L] fp32
        Ui = U_true_raw.imag.float().permute(0, 2, 1)
        Ur_s = F.conv1d(Ur, self.simpson_kernel.float(), padding=2, groups=D)[..., :-2].float()
        Ui_s = F.conv1d(Ui, self.simpson_kernel.float(), padding=2, groups=D)[..., :-2].float()
        U_true      = torch.complex(Ur_s, Ui_s).permute(0, 2, 1)        # [B, L, D]
        U_freq_true = torch.fft.fft(U_true.permute(0, 2, 1), n=n_fft, dim=2)
        H           = torch.fft.ifft(
            U_freq_true * K_freq.unsqueeze(0), n=n_fft, dim=2
        )[:, :, :L].permute(0, 2, 1)                                     # [B, L, D]

        # Manifold projection at READ TIME ONLY — not inside the recurrence.
        # This keeps training (FFT) and inference (step loop) bit-identical:
        # both accumulate raw state, both project only when measuring.
        H_proj = torch.complex(F.layer_norm(H.real, [D]), F.layer_norm(H.imag, [D]))

        # Spectrometer: K bands each measure D/K modes independently.
        # Bands follow HiPPO frequency quartiles: band 0 = slowest modes
        # (long memory), band K-1 = fastest modes (reflexes).
        # α_k = ⟨h_k*, φ_k⟩ / (D/K)  — one complex amplitude per band.
        K  = self.n_bands
        bx = self.band_idx                                              # [K, D/K]
        Hb = H_proj[:, :, bx]                                          # [B, L, K, D/K]
        Pb = phi[:, :, bx]                                             # [B, L, K, D/K]
        alpha = (Hb.conj() * Pb).sum(-1, keepdim=True) / (D // K)     # [B, L, K, 1]

        # Global workspace ignition: bands compete via softmax sharpened by tau.
        # tau=1 at init → soft competition. Sharpens as bands specialise.
        # Phase preserved — only magnitude |alpha| determines the winner.
        # *K restores total energy (softmax sums to 1, not K).
        c_k      = F.softmax(alpha.abs() / self.tau.clamp(min=1e-4), dim=2)  # [B, L, K, 1]
        alpha_ig = alpha * c_k * K                                            # [B, L, K, 1]

        # Self-model: per-band Born-rule density |h_k|² / (D/K).
        # Each band reports its own self-confidence, not the global field.
        # beta=0 at init → pure external measurement. Activates when density
        # carries useful gradient signal (i.e. confident bands speak louder).
        density   = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)          # [B, L, K, 1]
        alpha_fin = alpha_ig + self.beta * density                             # [B, L, K, 1]

        out_bands = (Hb * alpha_fin).real                                      # [B, L, K, D/K]
        out = torch.zeros(B, L, D, device=x.device)
        out[:, :, bx] = out_bands
        # ── MIMO: grouped cross-channel mixing ───────────────────────────────
        # [B, L, D] → [B, L, G, P] → P×P matmul per group → [B, L, D]
        # Cross-channel coupling within each group; ~2P× more FLOPs than SSM alone.
        G, P = D // self.mimo_p, self.mimo_p
        out = torch.einsum('gpq,blgq->blgp', self.mimo_w, out.reshape(B, L, G, P)).reshape(B, L, D)
        return out                                                              # [B, L, D]

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
        D = self.dim

        psi_raw = self.to_psi(x).float()
        psi     = torch.complex(psi_raw[..., :D], psi_raw[..., D:])    # [B, D]
        phi_raw = self.to_phi(x).float()
        phi     = torch.complex(phi_raw[..., :D], phi_raw[..., D:])    # [B, D]

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
        B_c_scout    = torch.complex(b_scout, torch.zeros_like(b_scout))
        h_next_scout = h_prev_scout * self._A() + gate * B_c_scout * psi  # [B, D]

        # 2. True step — Simpson's rule: S(U)_t = (1/6)U_t + (4/6)U_{t-1} + (1/6)U_{t-2}
        h_prev_flat  = torch.cat([h_prev_scout.real, h_prev_scout.imag], dim=-1)  # [B, 2D]
        b_true       = F.softplus(self.to_B_true_x(x).float() + self.to_B_true_h(h_prev_flat).float())
        B_c_true     = torch.complex(b_true, torch.zeros_like(b_true))
        U_curr       = gate * B_c_true * psi                              # [B, D] raw U
        U_smooth     = (1/6)*U_curr + (4/6)*h_prev_U1 + (1/6)*h_prev_U2  # Simpson's
        h_next_true  = h_prev_true * self._A() + U_smooth                 # [B, D]

        h_next = torch.stack([h_next_scout, h_next_true, U_curr, h_prev_U1], dim=1)  # [B, 4, D]

        # Projection at READ TIME ONLY — project the true wave for measurement.
        h_read = torch.complex(F.layer_norm(h_next_true.real, [D]),
                               F.layer_norm(h_next_true.imag, [D]))

        # Spectrometer + ignition + self-model — bit-identical to forward()
        Bsz = x.shape[0]
        K   = self.n_bands
        bx  = self.band_idx
        Hb  = h_read[:, bx]                                            # [B, K, D/K]
        Pb  = phi[:, bx]
        alpha    = (Hb.conj() * Pb).sum(-1, keepdim=True) / (D // K)  # [B, K, 1]
        c_k      = F.softmax(alpha.abs() / self.tau.clamp(min=1e-4), dim=1)  # [B, K, 1]
        alpha_ig = alpha * c_k * K
        density  = (Hb.real**2 + Hb.imag**2).mean(-1, keepdim=True)   # [B, K, 1]
        alpha_fin = alpha_ig + self.beta * density
        out_bands = (Hb * alpha_fin).real
        out = torch.zeros(Bsz, D, device=x.device)
        out[:, bx] = out_bands
        # MIMO grouped mixing — matches forward() exactly
        G, P = D // self.mimo_p, self.mimo_p
        out = torch.einsum('gpq,bgq->bgp', self.mimo_w, out.reshape(Bsz, G, P)).reshape(Bsz, D)
        return out, h_next


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
                 use_checkpoint: bool = True):
        super().__init__()
        self.dim            = dim
        self.depth          = depth
        self.use_checkpoint = use_checkpoint

        self.embed     = nn.Embedding(vocab, dim)
        # Independent physics per layer: each has its own omega, log_gamma, dt.
        # Layer 0 tends to learn letter-level patterns (fast decay).
        # Layer 5 tends to learn theme-level patterns (slow decay).
        self.operators = nn.ModuleList([SchrodingerAttention(dim, mimo_p=MIMO_P) for _ in range(depth)])
        self.depth_emb = nn.Parameter(torch.randn(depth, dim) * 0.02)
        self.out_proj  = nn.Linear(dim, vocab)

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
            z_prev = z   # this layer's output = next layer's prediction
        logits = self.out_proj(z)                                       # [B, L, V]
        return logits, z

    def forward_step(self, x: torch.Tensor, h_prev: torch.Tensor):
        """
        Single-token recurrent step.

        x      : [B, 1]       token indices
        h_prev : [B, depth, 2, D]  complex — [:, i, 0] = scout, [:, i, 1] = true
        returns: logits [B, 1, V], h_new [B, depth, 2, D]
        """
        z        = self.embed(x).squeeze(1)                            # [B, D]
        z_prev   = torch.zeros_like(z)
        h_layers = []
        for i in range(self.depth):
            z_in = z + self.depth_emb[i]
            z, h_i = self.operators[i].forward_step(z_in, h_prev[:, i], z_prev=z_prev)
            z_prev = z
            h_layers.append(h_i)
        h_new  = torch.stack(h_layers, dim=1)                          # [B, depth, 2, D]
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
    h = torch.zeros(1, model.depth, 4, model.dim, dtype=torch.complex64, device=device)

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
            h = torch.zeros(1, model.depth, 3, model.dim,
                            dtype=torch.complex64, device=device)
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

    data_path  = prepare_openhermes_dataset()
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

    opt      = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
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
                    ema_loss    = 0.95 * ema_loss + 0.05 * loss.item()
                    stability   = 1.0 / (grad_norm.item() + 1e-6)
                    reactive_lr = BASE_LR / (1.0 + np.exp(-(stability - 0.5)))
                    for pg in opt.param_groups:
                        pg["lr"] = reactive_lr

                opt.step()

                if step % PRINT_EVERY == 0:
                    acc = (logits.argmax(-1) == y).float().mean()
                    dts = " ".join(f"{op.dt.item():.3f}" for op in model.operators)
                    print(
                        f"Step {step:05d} | L={x.shape[1]:4d} | Loss: {loss.item():.4f} | "
                        f"Acc: {acc.item()*100:.1f}% | LR: {reactive_lr:.2e} | "
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
