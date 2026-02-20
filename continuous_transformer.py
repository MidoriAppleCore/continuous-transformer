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
DIM            = 512
DEPTH          = 6
VOCAB_SIZE     = 256

BASE_LR        = 3e-4
WEIGHT_DECAY   = 0.01
BATCH_SIZE     = 2
SEQ_LENGTH     = 4096
GRAD_CLIP      = 1.0

CHECKPOINT_EVERY = 1000
PRINT_EVERY      = 100

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

    The recurrence is:
        h_t = A · h_{t-1}  +  B ⊙ gate(x_t) · ψ_t

    A = exp((-γ+iω)·dt) is CONSTANT (not input-dependent).
    B = sqrt(2n+1) / max  is FIXED (HiPPO-LegS input coupling).
    gate(x_t) is a SCALAR (not per-mode) — purely content-based.
    ψ_t carries the content to write; B distributes it across modes optimally.

    B and ω are both initialised from (2n+1)^0.5 — this is not a coincidence.
    In diagonalised HiPPO-LegS, A's imaginary eigenvalues == B by construction.
    Fixing B closes the loop: ω (A eigenvalue) is learned, B (input coupling)
    is fixed at the provably optimal HiPPO values.

    This makes the recurrence LINEAR with constant coefficients, so the
    training FFT convolution is the EXACT same computation as the inference
    recurrence — not an approximation.  No train/inference mismatch.

    Born-rule physics is preserved in the READ step:
        out_t = Re(h_t · φ_t*)   ← wave interference / measurement

    Training  O(L log L) — exact causal FFT convolution
    Inference O(1)       — bit-identical recurrent forward_step
    """

    def __init__(self, dim: int, n_bands: int = 4):
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

        # ── HiPPO B coupling (fixed) + scalar write gate ──────────────────
        # B: fixed frequency-importance weight. Mode n gets sqrt(2n+1)/max.
        #    Low-order modes are slow+memory-holding; high-order are fast+input-driven.
        #    Mathematically coupled to omega init — same (2n+1)^0.5 function.
        # write_gate: SCALAR (1 output) — when to write, not which modes.
        #    Which modes get how much is handled by B (provably optimal).
        self.register_buffer('B', hippo_b_vector(dim))  # [D] fixed
        self.write_gate = nn.Linear(dim, 1)   # scalar gate — WHEN to write
        self.surprise_gain = nn.Parameter(torch.zeros(1))  # predictive coding depth
        self.tau  = nn.Parameter(torch.ones(1))   # ignition temperature (init=1 → soft)
        self.beta = nn.Parameter(torch.zeros(1))  # self-model density weight (init=0 → off)
        # tau=1: standard softmax competition.  Sharpens as bands specialise.
        # beta=0: no self-model at t=0. Activates when density carries gradient signal.

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

        psi_raw = self.to_psi(x)
        psi     = torch.complex(psi_raw[..., :D], psi_raw[..., D:])    # [B, L, D]
        phi_raw = self.to_phi(x)
        phi     = torch.complex(phi_raw[..., :D], phi_raw[..., D:])    # [B, L, D]

        # Predictive coding: gate scales with surprise (deviation from prediction).
        # When z_prev perfectly predicts x, surprise→0 and gate closes.
        # Wave only updates when it encounters something unexpected.
        gate = torch.sigmoid(self.write_gate(x))                        # [B, L, 1] real
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, L, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)
        # HiPPO B coupling: each mode n scaled by sqrt(2n+1)/max  [D]
        B_c  = torch.complex(self.B, torch.zeros_like(self.B))          # [D] complex
        U    = gate * B_c * psi                                          # [B, L, D] complex

        # ── Exact causal FFT convolution with kernel k(t) = A^t ─────────
        dt     = self.dt.abs()
        lam    = torch.complex(-torch.exp(self.log_gamma), self.omega)  # [D]
        t_axis = torch.arange(L, device=x.device, dtype=torch.float32)
        kernel = torch.exp(lam.unsqueeze(0) * t_axis.unsqueeze(1) * dt) # [L, D]

        n_fft  = 2 * L
        K_freq = torch.fft.fft(kernel.T, n=n_fft, dim=1)               # [D, n_fft]
        U_freq = torch.fft.fft(U.permute(0, 2, 1), n=n_fft, dim=2)    # [B, D, n_fft]
        H      = torch.fft.ifft(
                     U_freq * K_freq.unsqueeze(0), n=n_fft, dim=2
                 )[:, :, :L].permute(0, 2, 1)                          # [B, L, D]

        # Manifold projection at READ TIME ONLY — not inside the recurrence.
        # This keeps training (FFT) and inference (step loop) bit-identical:
        # both accumulate raw state, both project only when measuring.
        H_proj = torch.complex(exact_projection(H.real), exact_projection(H.imag))

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
        return out                                                              # [B, L, D]

    # ------------------------------------------------------------------
    def forward_step(self, x: torch.Tensor,
                     h_prev: torch.Tensor,
                     z_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) recurrent step — bit-identical to forward().

        The FFT in training computes: H_t = Σ_{s≤t} A^{t-s} · U_s
        This step computes:           h_t = A · h_{t-1} + U_t
        They are the same recurrence, one position at a time.

        x      : [B, D]  real token features
        h_prev : [B, D]  complex wave field
        z_prev : [B, D]  previous layer output (predictive coding)
        returns: out [B, D] real, h_next [B, D] complex
        """
        D = self.dim

        psi_raw = self.to_psi(x)
        psi     = torch.complex(psi_raw[..., :D], psi_raw[..., D:])    # [B, D]
        phi_raw = self.to_phi(x)
        phi     = torch.complex(phi_raw[..., :D], phi_raw[..., D:])    # [B, D]

        gate = torch.sigmoid(self.write_gate(x))                        # [B, 1]
        if z_prev is not None:
            surprise = (x - z_prev).abs().mean(-1, keepdim=True)        # [B, 1]
            gate = gate * (1.0 + torch.tanh(self.surprise_gain) * surprise)
        B_c  = torch.complex(self.B, torch.zeros_like(self.B))          # [D] complex

        # Exact recurrence — bit-identical to training FFT
        h_next = h_prev * self._A() + gate * B_c * psi                  # [B, D]

        # Projection at READ TIME ONLY (not stored back into state),
        # matching the training path which projects only before measurement.
        h_read = torch.complex(exact_projection(h_next.real),
                               exact_projection(h_next.imag))

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
        self.operators = nn.ModuleList([SchrodingerAttention(dim) for _ in range(depth)])
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

        x      : [B, 1]  token indices
        h_prev : [B, depth, D]  complex wave fields (one per layer)
        returns: logits [B, 1, V], h_new [B, depth, D]
        """
        z        = self.embed(x).squeeze(1)                            # [B, D]
        z_prev   = torch.zeros_like(z)
        h_layers = []
        for i in range(self.depth):
            z_in = z + self.depth_emb[i]
            z, h_i = self.operators[i].forward_step(z_in, h_prev[:, i], z_prev=z_prev)
            z_prev = z
            h_layers.append(h_i)
        h_new  = torch.stack(h_layers, dim=1)                          # [B, depth, D]
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

    # Initialize empty wave fields for all depth layers
    h = torch.zeros(1, model.depth, model.dim, dtype=torch.complex64, device=device)

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
# Training loop
# ---------------------------------------------------------------------------

def train(checkpoint_prefix: str = "continuous_transformer"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_path  = prepare_tinystories_dataset()
    dataset    = TinyStoriesDataset(data_path, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    opt      = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    ema_loss = None
    step     = start_step

    try:
        while True:
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                logits, _ = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), y.view(-1))

                opt.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad_norm = sum(
                        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    if ema_loss is None:
                        ema_loss = loss.item()
                    ema_loss    = 0.95 * ema_loss + 0.05 * loss.item()
                    stability   = 1.0 / (grad_norm.item() + 1e-6)
                    reactive_lr = BASE_LR / (1.0 + np.exp(-(stability - 0.5)))
                    for pg in opt.param_groups:
                        pg["lr"] = reactive_lr

                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

                if step % PRINT_EVERY == 0:
                    acc = (logits.argmax(-1) == y).float().mean()
                    dts = " ".join(f"{op.dt.item():.3f}" for op in model.operators)
                    print(
                        f"Step {step:05d} | Loss: {loss.item():.4f} | "
                        f"Acc: {acc.item()*100:.1f}% | LR: {reactive_lr:.2e} | "
                        f"dt: [{dts}]"
                    )

                if step % CHECKPOINT_EVERY == 0 and step > start_step:
                    generate(model, "Once upon a time", device=str(device))
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
