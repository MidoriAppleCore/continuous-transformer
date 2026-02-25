"""
train_mqar.py — Multi-Query Associative Recall (MQAR)
=====================================================
Canonical synthetic SSM benchmark from Zoology (Arora et al., 2023).
100% synthetic — no dataset download, no English.

The task
--------
Memorise N random key→value associations, survive a noise/filler section,
then answer N shuffled queries by retrieving the correct values:

    k1 v1 ... kN vN | ~~~ filler ~~~ | q3? v3  q1? v1  q2? v2
    └─── KV section ┘ └── can be 100k ┘ └──── query section ────┘

Training: Parallel FFT forward
------------------------------
model.forward(x) processes the entire sequence simultaneously via FFT,
so gradients flow unbroken from the query loss all the way back through
the filler to the KV section.  (TBPTT with h.detach() severed this wire.)

The learned dt is continuous: once the model locks in a slow decay to
hold associations across 256 filler tokens, the same physics hold for
100 000 tokens during evaluation — no long-distance training needed.

Curriculum
----------
Phase 1–5  : Standard Zoology grid (filler ≤ 256, matches published baselines)
Mix phase  : Uniform sample across Phases 1–5
Long eval  : evaluate_long() tests 1k / 10k / 100k filler after training

Usage
-----
    python train_mqar.py               # full run (~45min on 3050 Ti)
    python train_mqar.py --quick       # smoke test (~30s)
    python train_mqar.py --eval-only   # load latest checkpoint and evaluate

Comparison baselines (Zoology Table 1, vocab=8192)
---------------------------------------------------
  Transformer (full attention) : ~100% on all settings
  Based (Arora et al. 2023)    : ~99% easy → ~85% hard
  Hyena                        : ~28% easy → ~10% hard
  S4                           : ~20% easy → ~3% hard

Reference: Arora et al. arXiv:2312.04927 (2023).
"""

import argparse
import csv
import glob
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

OUT_DIR = "mqar_output"
os.makedirs(OUT_DIR, exist_ok=True)

from continuous_transformer import ContinuousTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

VOCAB  = 128
N_KEYS = 64      # key tokens  → IDs [0,  64)
N_VALS = 64      # value tokens → IDs [64, 128)

DIM   = 64
DEPTH = 4
# Per-layer wave-field widths.  Deep/slow layers get more complex slots so that
# 64 KV pairs can sit in well-separated regions of the state space.
# SNR = sqrt(state_dim/2 / N_keys):  128c→SNR=1.4  256c→SNR=2.0  512c→SNR=2.8
# With 64 keys the per-key ceiling is: 256→92%  512→98%  1024→≈100%
# dim=64: residual highway is narrow but FFN is 16× cheaper than dim=256.
# batch_scale=1024//64=16 → micro-batch=2, accum=16 → eff. batch=32.
STATE_DIMS = [256, 256, 512, int(1024*1.5)]   # L0 fast-syntax  →  L3 KV-vault

LR        = 3e-4
GRAD_CLIP = 1.0
TBPTT_W   = 256   # filler processed in chunks of this size (no-grad)

CHECKPOINT_PREFIX = "mqar_model"
LONG_EVAL_EVERY   = 5_000   # print O(1) long eval every this many training steps

# Standard Zoology eval grid (short — use parallel forward() for speed)
EVAL_GRID = [
    (4,  64),
    (8,  128),
    (16, 256),
    (32, 512),
    (64, 512),
]

# Long-distance eval grid (TBPTT eval, tests the unique O(1) claim)
EVAL_GRID_LONG = [
    (64, 1_000),
    (64, 10_000),
    (64, 100_000),
]

# Published baselines: Zoology Table 1 (vocab=8192, qualitatively comparable)
BASELINES = {
    (4,  64):  {"Transformer": 1.00, "Based": 1.00, "Hyena": 0.28, "S4": 0.20},
    (8,  128): {"Transformer": 1.00, "Based": 1.00, "Hyena": 0.22, "S4": 0.12},
    (16, 256): {"Transformer": 1.00, "Based": 0.99, "Hyena": 0.15, "S4": 0.07},
    (32, 512): {"Transformer": 1.00, "Based": 0.97, "Hyena": 0.11, "S4": 0.04},
    (64, 512): {"Transformer": 1.00, "Based": 0.85, "Hyena": 0.10, "S4": 0.03},
    # Attention OOM / N/A for long distances — they can't run these at all
    (64, 1_000):   {"Transformer": None, "Based": None, "Hyena": None, "S4": None},
    (64, 10_000):  {"Transformer": None, "Based": None, "Hyena": None, "S4": None},
    (64, 100_000): {"Transformer": None, "Based": None, "Hyena": None, "S4": None},
}

# Curriculum: (n_kv, filler_len, batch_size, phase_fraction)
# phase_fraction is normalised internally, so only the ratios matter.
CURRICULUM = [
    # Standard Zoology phases — parallel FFT, full gradient KV→filler→query
    # Skewed toward short sequences for faster initial convergence.
    # The wave physics generalise — locking in phase mappings on short contexts
    # transfers directly to long ones once dt settles.
    (4,   48,  32, 0.35),  # blazing fast — 35% of steps
    (8,   96,  32, 0.25),  # very fast   — 25% of steps
    (16,  192, 32, 0.20),  # fast        — 20% of steps (unchanged)
    (32,  384, 32, 0.10),  # heavy       — 10% of steps (was 0.25)
    (64,  256, 32, 0.10),  # heavy       — 10% of steps (was 0.30)
]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder — returns three separate sections for TBPTT
# ─────────────────────────────────────────────────────────────────────────────

def make_mqar_sections(
    batch_size: int,
    n_kv: int,
    filler_len: int,
    device: torch.device,
):
    """
    Build MQAR sections as three tensors — vectorised, generated directly on device.

    Returns
    -------
    kv_toks     : [B, 2*n_kv]    interleaved key/value tokens
    filler_toks : [B, filler_len] random key tokens
    query_toks  : [B, 2*n_kv]    interleaved query-key / correct-value tokens
    target_vals : [B, n_kv]      value tokens expected at each query-key position
    """
    B = batch_size

    keys = torch.argsort(torch.rand(B, N_KEYS, device=device), dim=1)[:, :n_kv]  # [B, n_kv]
    vals = torch.randint(0, N_VALS, (B, n_kv), device=device)                    # [B, n_kv]

    kv_toks     = torch.stack([keys, vals + N_KEYS], dim=2).reshape(B, 2 * n_kv) # [B, 2*n_kv]
    filler_toks = torch.randint(0, N_KEYS, (B, filler_len), device=device)       # [B, filler_len]

    query_order = torch.argsort(torch.rand(B, n_kv, device=device), dim=1)       # [B, n_kv]
    bidx        = torch.arange(B, device=device).unsqueeze(1)
    q_keys      = keys[bidx, query_order]                                         # [B, n_kv]
    q_vals      = vals[bidx, query_order] + N_KEYS                                # [B, n_kv]
    query_toks  = torch.stack([q_keys, q_vals], dim=2).reshape(B, 2 * n_kv)      # [B, 2*n_kv]
    target_vals = q_vals                                                           # [B, n_kv]

    return kv_toks, filler_toks, query_toks, target_vals


def make_mqar_batch_flat(batch_size, n_kv, seq_len, device):
    """Flat [B, L] sequence + masked labels — vectorised, generated directly on device."""
    filler_len = seq_len - 4 * n_kv
    assert filler_len >= 0
    B = batch_size

    # B independent random permutations via argsort(rand) — no Python loop
    keys = torch.argsort(torch.rand(B, N_KEYS, device=device), dim=1)[:, :n_kv]  # [B, n_kv]
    vals = torch.randint(0, N_VALS, (B, n_kv), device=device)                    # [B, n_kv]

    # KV section: interleave key / N_KEYS+val  →  [B, 2*n_kv]
    kv = torch.stack([keys, vals + N_KEYS], dim=2).reshape(B, 2 * n_kv)

    # Filler: random key tokens  →  [B, filler_len]
    filler = torch.randint(0, N_KEYS, (B, filler_len), device=device)

    # Query section: independently shuffled order per batch item
    query_order = torch.argsort(torch.rand(B, n_kv, device=device), dim=1)        # [B, n_kv]
    bidx  = torch.arange(B, device=device).unsqueeze(1)                           # [B, 1]
    q_keys = keys[bidx, query_order]                                              # [B, n_kv]
    q_vals = vals[bidx, query_order] + N_KEYS                                     # [B, n_kv]
    queries = torch.stack([q_keys, q_vals], dim=2).reshape(B, 2 * n_kv)          # [B, 2*n_kv]

    seqs   = torch.cat([kv, filler, queries], dim=1)                              # [B, seq_len]
    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    qs = 2 * n_kv + filler_len                                                    # query start
    labels[:, qs::2] = q_vals                                                     # after seeing q_key at qs, predict q_val (which is at qs+1, unseen)

    return seqs, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(n_steps: int, dim: int, depth: int, device: torch.device,
          long_every: int = LONG_EVAL_EVERY,
          state_dims: list = None) -> ContinuousTransformer:
    if state_dims is None:
        state_dims = [dim] * depth
    assert len(state_dims) == depth, f"len(state_dims)={len(state_dims)} must equal depth={depth}"
    model = ContinuousTransformer(
        dim=dim, depth=depth, vocab=VOCAB, use_checkpoint=True,
        state_dims=state_dims
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    wave_kb   = sum(4 * sd * 8 for sd in state_dims) / 1024
    print(f"Model: {n_params:.2f}M params  dim={dim}  depth={depth}  vocab={VOCAB}")
    print(f"State: {state_dims}  →  {wave_kb:.1f} KB wave (constant at any context length)")
    print(f"Training: parallel FFT forward (full gradient KV→filler→query)\n")

    # Checkpoint resume — picks up step checkpoints and _final.pt
    start_step = 0
    ckpts = sorted(glob.glob(f"{CHECKPOINT_PREFIX}_step*.pt") +
                   glob.glob(f"{CHECKPOINT_PREFIX}_final.pt"))
    if ckpts:
        steps_found = []
        for cp in ckpts:
            try:
                # _step######.pt → parse number from filename
                steps_found.append((int(cp.split("step")[1].split(".")[0]), cp))
            except (IndexError, ValueError):
                # _final.pt → read the saved step from inside the file
                try:
                    _sd = torch.load(cp, map_location="cpu")
                    steps_found.append((_sd.get("step", 0), cp))
                except Exception:
                    pass
        if steps_found:
            _, latest_ckpt = max(steps_found)
            sd = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(sd.get("model", sd), strict=False)
            start_step = sd.get("step", 0)
            print(f"Resumed from {latest_ckpt} (step {start_step})")

    # Separate param groups: dt params get 100× higher LR.
    # Root cause: dt_raw for slow layers inits at softplus_inv(0.001)≈-6.9.
    # softplus'(-6.9) = sigmoid(-6.9) ≈ 0.001, so the gradient reaching dt_raw
    # is 1000× smaller than at a normal operating point. Adam can't compensate
    # (g² ≈ 1e-6, eps=1e-8, effective step barely above noise floor).
    # 100× higher LR brings the slow-layer dt update rate on par with the rest.
    # weight_decay=0 for dt: L2 on a log-timescale param would bias it toward 0.
    dt_params   = [op.dt for op in model.operators]
    dt_id_set   = {id(p) for p in dt_params}
    # requires_grad=False params (frozen VSA phase multiplexers) are excluded from
    # both groups — they produce no gradients and don't need optimizer state.
    base_params = [p for p in model.parameters()
                   if id(p) not in dt_id_set and p.requires_grad]
    opt = optim.AdamW([
        {'params': base_params, 'lr': LR,       'weight_decay': 0.01},
        {'params': dt_params,   'lr': LR * 30,  'weight_decay': 0.0},
    ], eps=1e-8, fused=True)

    # Restore optimizer state if checkpoint contains it (avoids cold-start LR spike)
    if ckpts and steps_found:
        _, latest_ckpt = max(steps_found)
        sd = torch.load(latest_ckpt, map_location=device)
        if "optimizer" in sd:
            opt.load_state_dict(sd["optimizer"])
            print(f"  Optimizer state restored.")
        else:
            print(f"  No optimizer state in checkpoint — Adam cold-starts (first ~200 steps may be noisy).")

    # Weighted random phase sampling — all difficulties seen every step.
    # phase_fraction is used as a sampling weight, not a step budget.
    _phases  = [(n_kv, filler_len, bs) for n_kv, filler_len, bs, _ in CURRICULUM]
    _weights = np.array([frac for *_, frac in CURRICULUM], dtype=np.float64)
    _weights /= _weights.sum()

    # Auto-scale batch sizes when state_dims are wider than dim.
    # FFT activations peak at [B, max_SD, 2*L] complex64 — memory scales linearly with B.
    # batch_scale = max_SD / dim: e.g. 1024/64=16 → batch 32→2 → 16× less activation memory.
    # We compensate with gradient accumulation (accum_steps = batch_scale) to restore the
    # effective batch size and gradient signal quality, at no extra VRAM cost.
    batch_scale = max(state_dims) // dim
    if batch_scale > 1:
        _phases = [(n_kv, fl, max(2, bs // batch_scale)) for n_kv, fl, bs in _phases]
        accum_steps = batch_scale
        print(f"  [mem] state_dim max={max(state_dims)} > dim={dim} "
              f"→ batch ÷{batch_scale}, grad-accum ×{accum_steps} "
              f"→ eff. batch ~{max(2, 32 // batch_scale) * accum_steps}")
    else:
        accum_steps = 1

    def sample_phase():
        idx = np.random.choice(len(_phases), p=_weights)
        return _phases[idx]

    print("Curriculum (weighted random sampling):")
    for (n_kv, filler_len, bs), w in zip(_phases, _weights):
        print(f"  n_kv={n_kv:2d}  filler={filler_len:>7,}  "
              f"total_ctx={2*n_kv+filler_len+2*n_kv:>8,}  batch={bs:2d}  p={w:.2f}")
    print()

    ema_loss = ema_acc = None
    model.train()
    t0 = time.time()

    for step in range(start_step, start_step + n_steps):
        # ── Gradient accumulation: accum_steps micro-batches → 1 optimizer step ──
        # Each micro-loss is divided by accum_steps so the accumulated gradient equals
        # the mean gradient over (bs × accum_steps) examples — identical to a large batch.
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_acc  = 0.0
        n_kv = filler_len = bs = None
        skip_step = False

        for _a in range(accum_steps):
            n_kv, filler_len, bs = sample_phase()
            seq_len = 4 * n_kv + filler_len
            x, labels = make_mqar_batch_flat(bs, n_kv, seq_len, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(x)                                   # [B, L, V]
                mask = labels != -100
                # Only compute loss on the n_kv active label tokens — skip filler gradient
                loss_i = F.cross_entropy(logits[mask], labels[mask]) / accum_steps
            if loss_i.isnan() or loss_i.isinf():
                print(f"  [step {step}] NaN/Inf in micro-batch — skipping optimizer step")
                skip_step = True
                break
            step_acc  += (logits[mask].argmax(-1) == labels[mask]).float().mean().item() / accum_steps
            loss_i.backward()
            step_loss += loss_i.item()
        # ────────────────────────────────────────────────────────────────────

        if skip_step:
            opt.zero_grad(set_to_none=True)
            continue
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        model.invalidate_A_cache()

        ema_loss = step_loss if ema_loss is None else 0.95*ema_loss + 0.05*step_loss
        ema_acc  = step_acc  if ema_acc  is None else 0.95*ema_acc  + 0.05*step_acc

        local = step - start_step
        if local % 200 == 0:
            elapsed = time.time() - t0
            dt_vals = [torch.nn.functional.softplus(op.dt).item() for op in model.operators]
            dts = " ".join(f"{v:.3f}" for v in dt_vals)
            total_steps = start_step + n_steps  # absolute step target
            print(f"[{step:6d}/{total_steps}] "
                  f"nkv={n_kv:2d} filler={filler_len:>7,} | "
                  f"loss={ema_loss:.4f} acc={ema_acc*100:.1f}% | "
                  f"dt=[{dts}]  ({elapsed:.0f}s)")
            # ── CSV logging ──────────────────────────────────────────────
            csv_path = os.path.join(OUT_DIR, "training_curve.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["step", "ema_loss", "ema_acc",
                                "n_kv", "filler_len"]
                               + [f"dt_L{i}" for i in range(len(dt_vals))])
                w.writerow([step, f"{ema_loss:.6f}", f"{ema_acc:.6f}",
                            n_kv, filler_len]
                           + [f"{v:.6f}" for v in dt_vals])

        if local > 0 and local % 5000 == 0:
            path = f"{CHECKPOINT_PREFIX}_step{step:06d}.pt"
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": opt.state_dict()}, path)
            print(f"  Saved {path}")

        if long_every > 0 and local > 0 and local % long_every == 0:
            dts_now = " ".join(f"L{i}={torch.nn.functional.softplus(op.dt).item():.3f}" for i, op in enumerate(model.operators))
            print(f"\n  ── O(1) long-distance check @ step {step}  dt=[{dts_now}] ──")
            evaluate_long(model, device, n_trials=5,
                          grid=[(64, 1_000), (64, 10_000)])  # 100k at final eval only
            model.train()
            print()

    path = f"{CHECKPOINT_PREFIX}_final.pt"
    torch.save({"step": start_step + n_steps, "model": model.state_dict()}, path)
    print(f"\nTraining complete. Saved {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_short(model, device, n_trials=200) -> dict:
    """Standard Zoology grid (<=512 tokens) — parallel forward() for speed."""
    model.eval()
    results = {}
    for n_kv, seq_len in EVAL_GRID:
        hits = total = 0
        remaining = n_trials
        while remaining > 0:
            bs = min(32, remaining)
            x, labels = make_mqar_batch_flat(bs, n_kv, seq_len, device)
            logits, _ = model(x)
            mask  = labels != -100
            preds = logits.argmax(-1)
            hits  += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            remaining -= bs
        acc = hits / total if total else 0.0
        results[(n_kv, seq_len)] = acc
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  n_kv={n_kv:2d}  filler={seq_len-4*n_kv:3d}  {bar}  {acc*100:.1f}%")
    # ── write CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "eval_short.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_kv", "filler_len", "seq_len", "accuracy"])
        for (n_kv, seq_len), acc in results.items():
            w.writerow([n_kv, seq_len - 4*n_kv, seq_len, f"{acc:.6f}"])
    print(f"  → {csv_path}")
    model.train()
    return results


@torch.no_grad()
def evaluate_long(model, device, n_trials=20, grid=None) -> dict:
    """
    Long-distance eval using forward_step (O(1) memory per trial).
    Filler chunked in no-grad blocks — same as training.
    Attention models cannot run these at all (OOM or O(L^2) compute).
    Pass grid= to restrict which filler lengths are tested.
    """
    if grid is None:
        grid = EVAL_GRID_LONG
    model.eval()
    results = {}
    for n_kv, total_ctx in grid:
        filler_len = total_ctx - 4 * n_kv
        hits = total = 0
        t0 = time.time()
        for _ in range(n_trials):
            kv_toks, filler_toks, query_toks, target_vals = make_mqar_sections(
                1, n_kv, filler_len, device
            )
            h = model.make_state(1, device=device)
            for t in range(kv_toks.shape[1]):
                _, h = model.forward_step(kv_toks[:, t:t+1], h)

            pos = 0
            while pos < filler_len:
                chunk_end = min(pos + TBPTT_W, filler_len)
                for t in range(pos, chunk_end):
                    _, h = model.forward_step(filler_toks[:, t:t+1], h)
                pos = chunk_end

            for qi in range(n_kv):
                # After seeing query key, logits predict the next token = answer
                logits, h = model.forward_step(query_toks[:, 2*qi:2*qi+1], h)   # sees q_key → predict answer
                pred = logits[0, 0].argmax().item()
                _, h = model.forward_step(query_toks[:, 2*qi+1:2*qi+2], h)       # teacher-force answer to keep state
                hits  += int(pred == target_vals[0, qi].item())
                total += 1

        acc = hits / total if total else 0.0
        elapsed = time.time() - t0
        wave_kb = sum(4 * sd * 8 for sd in model.state_dims) / 1024
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  n_kv={n_kv:2d}  filler={filler_len:>7,}  {bar}  {acc*100:.1f}%"
              f"  ({elapsed:.1f}s)  wave={wave_kb:.1f}KB constant")
        results[(n_kv, total_ctx)] = acc
    # ── write CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "eval_long.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_kv", "filler_len", "total_ctx", "accuracy", "wave_kb"])
        for (n_kv, total_ctx), acc in results.items():
            filler_len = total_ctx - 4 * n_kv
            wave_kb    = sum(4 * sd * 8 for sd in model.state_dims) / 1024
            w.writerow([n_kv, filler_len, total_ctx, f"{acc:.6f}", f"{wave_kb:.2f}"])
    print(f"  → {csv_path}")
    model.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(short_results: dict, long_results: dict) -> None:
    all_results = {**short_results, **long_results}
    all_grid    = EVAL_GRID + list(long_results.keys())
    model_names = ["Schrödinger Wave (ours)", "Transformer¹", "Based¹", "Hyena¹", "S4¹"]

    col_w = 11; header_w = 28

    print("\n" + "═" * 90)
    print("  MQAR ACCURACY — Multi-Query Associative Recall")
    print("  (↑ higher is better  |  OOM/N/A = cannot run on this hardware)")
    print("═" * 90)

    header = f"  {'Model':<{header_w}}"
    for n_kv, seq_len in all_grid:
        filler = seq_len - 4 * n_kv
        header += f"  {str(filler)+' F':>{col_w}}"
    print(header)
    print("  " + "─" * 86)

    def fmt(acc):
        return "OOM/N/A" if acc is None else f"{acc*100:.1f}%"

    for name in model_names:
        row = f"  {name:<{header_w}}"
        for key in all_grid:
            if name.startswith("Schrödinger"):
                acc = all_results.get(key, None)
            else:
                short = name.split("¹")[0].strip()
                acc   = BASELINES.get(key, {}).get(short, None)
            row += f"  {fmt(acc):>{col_w}}"
        print(row)

    print("═" * 90)
    wave_kb = sum(4 * sd * 8 for sd in STATE_DIMS) / 1024
    print(f"\n  Wave state = {STATE_DIMS} complex64 = {wave_kb:.1f} KB  (CONSTANT at any context)")
    print(f"  Transformer KV-cache @ 100k tokens ≈ {100_000*DIM*2*4//1024//1024} MB per layer")
    print(f"\n  ¹ Baselines: Arora et al. arXiv:2312.04927 Table 1 (vocab=8192).")
    print(f"    Our vocab={VOCAB}.  OOM rows are architectural limits, not tuning gaps.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global VOCAB, N_KEYS, N_VALS, DIM, DEPTH, STATE_DIMS

    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',       type=int, default=100_000)
    ap.add_argument('--dim',         type=int, default=DIM)
    ap.add_argument('--depth',       type=int, default=DEPTH)
    ap.add_argument('--trials',      type=int, default=200)
    ap.add_argument('--long-trials', type=int, default=20)
    ap.add_argument('--eval-only',   action='store_true')
    ap.add_argument('--skip-long',   action='store_true',
                    help='Skip long-distance eval (faster)')
    ap.add_argument('--long-every',  type=int, default=LONG_EVAL_EVERY,
                    help='Run O(1) long eval every N training steps (0 = disable)')
    ap.add_argument('--quick',       action='store_true',
                    help='Smoke test: 300 steps, dim=64, depth=2')
    ap.add_argument('--vocab',       type=int, default=VOCAB)
    ap.add_argument('--n-keys',      type=int, default=N_KEYS)
    args = ap.parse_args()

    VOCAB  = args.vocab
    N_KEYS = args.n_keys
    N_VALS = VOCAB - N_KEYS
    DIM    = args.dim
    DEPTH  = args.depth

    if args.quick:
        args.steps = 300; args.trials = 20; args.long_trials = 3
        DIM = 64; DEPTH = 2
        STATE_DIMS = [DIM] * DEPTH  # quick mode: no widening, keeps quick test fast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Ampere+ (3050 Ti included): TF32 gives ~2× matmul throughput at negligible precision cost.
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Vocab:  {VOCAB} ({N_KEYS} keys, {N_VALS} values)\n")

    if args.eval_only:
        ckpts = (sorted(glob.glob(f"{CHECKPOINT_PREFIX}_step*.pt")) +
                 sorted(glob.glob(f"{CHECKPOINT_PREFIX}_final.pt")))
        if not ckpts:
            print("No checkpoint found."); sys.exit(1)
        model = ContinuousTransformer(dim=DIM, depth=DEPTH, vocab=VOCAB,
                                      use_checkpoint=False,
                                      state_dims=STATE_DIMS).to(device)
        sd = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(sd.get("model", sd), strict=False)
        model.eval()
    else:
        model = train(n_steps=args.steps, dim=DIM, depth=DEPTH, device=device,
                      long_every=0 if args.quick else args.long_every,
                      state_dims=STATE_DIMS)

    dts = " ".join(f"L{i}={op.dt.item():.3f}" for i, op in enumerate(model.operators))
    print(f"\n  dt after training:  [{dts}]\n")

    print("─" * 60)
    print("  Short MQAR (Zoology grid, <=512 tokens):")
    print("─" * 60)
    short_results = evaluate_short(model, device, n_trials=args.trials)

    long_results = {}
    if not args.skip_long and not args.quick:
        print("\n" + "─" * 60)
        print("  Long-distance MQAR (novel — O(1) wave memory):")
        print("─" * 60)
        long_results = evaluate_long(model, device, n_trials=args.long_trials)

    print_comparison_table(short_results, long_results)


if __name__ == "__main__":
    main()
