"""
CakeNET — Schrödinger Wave Agent
=================================
The wave field h is the agent's only memory. It is never reset.

Each frame: 16 CNN patch tokens are injected one-by-one through the wave
layers (identical to how GPT-4V feeds image tokens into a language model).
After 16 steps h encodes the full visual scene. The agent then reads the
wave twice simultaneously — once for action, once for value — from different
physical timescales:

  Actor  → early layers  (fast dt, high-frequency → reflexes)
  Critic → deep  layers  (slow dt, low-frequency  → navigation intuition)

The wave's dt parameters learn which layers are fast and which are slow.
Actor/critic readout weights learn to chase those timescales.
Both self-organise toward the same solution from different directions.

Training: online n-step A2C, TBPTT-1 at the frame boundary.
  n=128 so the 5.0 delivery reward propagates through a full search cycle:
  cube spawns at radius 2-5 (up to 10 units away), 0.2 u/step → ~50 steps
  to reach, ~20 more to deliver.  n=128 gives 2× margin on worst case.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

from continuous_transformer import ContinuousTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


# ---------------------------------------------------------------------------
# Vision  64×64 RGB → 16 spatial tokens
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    CNN stem → 4×4 spatial grid → 16 tokens of dim D.

    Spatial structure is preserved: each token knows WHERE it is via a
    learnable 2-D positional embedding.  LayerNorm stabilises the output
    before it enters the complex wave field.

      64 → Conv(s=2) → 32 → Conv(s=2) → 16 → Conv(s=4) → 4×4 = 16 tokens
    """
    def __init__(self, dim=256, in_channels=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1),   # 64→32
            nn.GELU(),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),   # 32→16
            nn.GELU(),
            nn.Conv2d(64,  dim, 4, stride=4),              # 16→4
            nn.GELU(),
        )
        # Horizontal fibers: patches talk to their 4 spatial neighbours.
        # groups=dim → each feature channel mixes independently (depthwise).
        # Zero-init: starts as identity, learns spatial communication gradually.
        self.spatial_mix = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        nn.init.zeros_(self.spatial_mix.weight)
        nn.init.zeros_(self.spatial_mix.bias)
        self.pos_emb = nn.Parameter(torch.zeros(1, 16, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.norm = nn.LayerNorm(dim)
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        # Accept both uint8 [0,255] (Gym) and float [0,1] (Ursina/DummyWorld).
        # Both map to [-1, 1] — the wave physics expects zero-centred input.
        img = img.float()
        if img.max() > 1.5:          # uint8 path: [0,255] → [-1,1]
            img = img / 127.5 - 1.0
        else:                         # float path: [0,1] → [-1,1]
            img = img * 2.0 - 1.0
        feat   = self.cnn(img)                          # [B, D, 4, 4]
        feat   = feat + self.spatial_mix(feat)          # horizontal fibers: neighbours talk
        tokens = feat.flatten(2).transpose(1, 2)        # [B, 16, D]
        return self.norm(tokens + self.pos_emb)         # [B, 16, D]


# ---------------------------------------------------------------------------
# Top-down gate  — breaks the CNN wall
# ---------------------------------------------------------------------------

class TopDownGate(nn.Module):
    """
    The CNN wall, drilled through.

    The deepest True wave h[-1, slot=1] has integrated the full temporal
    history of the run into a complex oscillation:

        h = A·e^{iφ}     A = amplitude (what the wave has learned to expect)
                         φ = phase     (where in the processing cycle)

    We use this to reshape the CURRENT frame's tokens before they enter
    the wave — exactly as PFC/IT feedback reshapes V1 receptive fields
    during attention in a living brain.

    Re(h)  → spatial attention  — amplitude says WHERE to look.
             "I expect motion top-left" → boost those 4 patches.

    Im(h)  → FiLM modulation    — phase says WHAT to amplify.
             "I am in a retrieval phase" → scale+shift feature channels.

    Frame 0: h is all zeros → no modulation → pure bottom-up.
    As the wave charges up, top-down feedback gradually switches on.
    The wall doesn't exist anymore.
    """
    def __init__(self, dim: int):
        super().__init__()
        # FiLM parameters driven by wave phase (imaginary part)
        self.film_scale = nn.Linear(dim, dim)
        self.film_shift = nn.Linear(dim, dim)
        # Init: identity transform — zero top-down influence at birth
        nn.init.zeros_(self.film_scale.weight)
        nn.init.ones_ (self.film_scale.bias)   # scale = 1  (pass through)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)   # shift = 0  (no bias)

    def forward(self, patches: torch.Tensor,
                h_prev:  torch.Tensor) -> torch.Tensor:
        """
        patches : [B, 16, D]        bottom-up CNN tokens
        h_prev  : [B, depth, 4, D]  complex — previous frame's full wave
        returns : [B, 16, D]        top-down modulated tokens

        Spatial attention:
            α_n = softmax_n ( Re(h_deep) · p_n  /  √D )
            — the dot product of the wave's amplitude vector with each
              patch token. A patch resonates if its features align with
              what the wave has been predicting.

        Feature modulation (FiLM):
            out = patches + α ⊗ ( patches · scale(Im h) + shift(Im h) − patches )
            — phase gates WHICH features get amplified at each location.
              The residual form keeps bottom-up intact; top-down only adds.
        """
        D = patches.shape[-1]

        # First frame: wave is zero → nothing to attend to → pass through
        if all(t.abs().max() < 1e-6 for t in h_prev):
            return patches

        h_wave   = h_prev[-1][:, 1, :]                     # [B, state_dim_last] complex
        h_deep_r = self.state_proj(h_wave.real)             # [B, D] — projected to model dim
        h_deep_i = self.state_proj(h_wave.imag)             # [B, D]

        # — WHERE to look: complex amplitude as spatial query —
        # Re(h) ≡ the real-amplitude oscillation component that has accumulated
        # over all past frames.  Inner product with patches = resonance score.
        attn = F.softmax(
            torch.einsum('bd,bnd->bn', h_deep_r, patches) / (D ** 0.5),
            dim=-1
        )                                                  # [B, 16]

        # — WHAT to amplify: phase as FiLM condition —
        # Im(h) ≡ quadrature component — 90° ahead of the real wave.
        # In biology: different oscillatory phases gate encoding vs retrieval.
        scale     = self.film_scale(h_deep_i)              # [B, D]
        shift     = self.film_shift(h_deep_i)              # [B, D]
        modulated = patches * scale.unsqueeze(1) + shift.unsqueeze(1)  # [B, 16, D]

        # Spatial attention blends modulated and raw: peaked attn → focused
        # top-down; flat attn → diffuse modulation across all patches.
        # Residual: bottom-up is never destroyed, only enriched.
        return patches + attn.unsqueeze(-1) * (modulated - patches)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SchrodingerAgent(nn.Module):
    """
    Image → wave field → continuous Gaussian policy + scalar value.

    encode_obs injects 16 patch tokens sequentially through every layer,
    capturing the per-layer output z on the final token pass.  This gives
    actor and critic independent access to every timescale in the wave.

    Readout weights are learnable but initialised with a physical prior:
      actor_layer_w  linspace(1→0): early-layer emphasis  (fast, reflexive)
      critic_layer_w linspace(0→1): late-layer  emphasis  (slow, predictive)

    Once dt diverges across layers the outputs differ → gradients flow into
    the readout weights → they migrate to wherever the wave actually learned
    the timescale split.  Physics and credit assignment converge together.
    """
    def __init__(self, dim=256, depth=6, n_actions=4, state_dims=None):
        super().__init__()
        self.dim   = dim
        self.depth = depth

        self.vision   = VisionEncoder(dim=dim, in_channels=6)
        self.vis_proj = orthogonal_init(nn.Linear(dim, dim), gain=1.0)

        self.transformer = ContinuousTransformer(
            dim=dim, depth=depth, vocab=256, use_checkpoint=False,
            state_dims=state_dims
        )
        # top_down created after transformer so it can read transformer.state_dims[-1]
        self.top_down = TopDownGate(dim=dim,
                                    last_state_dim=self.transformer.state_dims[-1])

        # Physical prior: actor reads fast layers, critic reads slow layers.
        # Gradient on actor_layer_w[i]  ∝  z_layers[i] − z_actor
        # → once dt spreads, outputs differ → weights start moving.
        self.actor_layer_w  = nn.Parameter(torch.linspace(1.0, 0.0, depth))
        self.critic_layer_w = nn.Parameter(torch.linspace(0.0, 1.0, depth))

        # LayerNorm before policy head only — critic was learning fine without it.
        # actor_norm stabilises policy gradient when wave activations are small
        # (cold h early in training). critic_norm caused gradient explosion.
        self.actor_norm  = nn.LayerNorm(dim)
        self.policy_head = orthogonal_init(nn.Linear(dim, n_actions), gain=0.01)
        self.value_head  = orthogonal_init(nn.Linear(dim, 1),         gain=0.1)
        # World model: predicts the True wave's next real state from current.
        # The True wave h_{t+1}.real = A·h_t.real + input — this head learns
        # the autonomous A·h_t part, i.e. the Koopman operator of the dynamics.
        # Zero input → pure wave prediction. Supervised by the actual next h.
        sd_last = self.transformer.state_dims[-1]
        self.patch_pred  = orthogonal_init(nn.Linear(sd_last, sd_last), gain=1.0)

    def encode_obs(self, obs, h):
        """
        16 patch tokens → sequential wave steps → per-layer z + new h.

        Each token is stepped through all depth layers, with z_prev_layer
        tracking the previous layer's output for inter-layer predictive coding
        — exactly mirroring ContinuousTransformer.forward_step.

        Only z_layers from the LAST token is kept: that's the wave's response
        to the complete visual scene after all 16 patches have been absorbed.
        """
        patches = self.vision(obs)                          # [B, 16, D]  bottom-up
        patches = self.top_down(patches, h)                 # [B, 16, D]  + top-down
        B = patches.shape[0]
        z_layers = [None] * self.depth   # will be overwritten each token step
        _zprev_buf = None   # allocated once on first token; reused across 16 tokens
        for tok_i in range(patches.shape[1]):
            token        = self.vis_proj(patches[:, tok_i])    # [B, D]
            z            = token
            if _zprev_buf is None:
                _zprev_buf = torch.zeros(B, self.dim, device=obs.device, dtype=token.dtype)
            z_prev_layer = _zprev_buf   # zeros for layer-0; rebinds to z inside loop
            h_layers     = []
            for i in range(self.depth):
                z, h_i = self.transformer.operators[i].forward_step(
                    z + self.transformer.depth_emb[i],
                    h[i],
                    z_prev=z_prev_layer,     # inter-layer predictive coding
                )
                z = self.transformer.ffn[i](z)  # SwiGLU non-linear decode (matches forward_step)
                z = z.clamp(-10, 10)         # prevent cascading amplification across layers
                z_prev_layer = z             # this layer's output predicts next layer
                z_layers[i]  = z
                h_layers.append(h_i)
            h = h_layers                                   # list[depth × [B, 4, state_dim_i]]
        return z_layers, h                                 # list[B,D]×depth, list[B,4,SD_i]×depth

    def forward(self, obs, h_prev):
        z_layers, h_next = self.encode_obs(obs, h_prev)
        z_all    = torch.stack(z_layers, dim=1)            # [B, depth, D]
        aw       = F.softmax(self.actor_layer_w,  dim=0)   # [depth]
        cw       = F.softmax(self.critic_layer_w, dim=0)   # [depth]

        # Late fusion: apply heads to every layer independently, then mix.
        # Gradient on actor_layer_w[k] ∝ (all_logits[:,k,:] - logits) — i.e.
        # how much layer k's action prediction differs from the weighted consensus.
        # In 4-D logit space small z differences produce large logit differences,
        # giving actor_layer_w a real gradient signal even when z_k are cosine-similar.
        # Previously we mixed in 256-D first: z_k ≈ z_actor → gradient ≈ 0 forever.
        all_logits = self.policy_head(self.actor_norm(z_all))  # [B, depth, n_actions]
        all_values = self.value_head(z_all)                    # [B, depth, 1]
        logits = (all_logits * aw.view(1, -1, 1)).sum(1)       # [B, n_actions]
        value  = (all_values * cw.view(1, -1, 1)).sum(1)       # [B, 1]
        return logits, value, h_next

    def trainable_params(self):
        """Exclude LM-only embed and out_proj — zero RL gradient reaches them."""
        dead = {'transformer.embed.weight',
                'transformer.out_proj.weight',
                'transformer.out_proj.bias'}
        return [p for n, p in self.named_parameters() if n not in dead]

    def init_hidden(self, device):
        # 4-slot state: scout, true, U_{t-1}, U_{t-2} — matches Simpson's forward_step
        return self.transformer.make_state(1, device=device)


# ---------------------------------------------------------------------------
# Dummy world  (--dummy flag, no Ursina needed)
# ---------------------------------------------------------------------------

class DummyWorld:
    score     = 0
    n_actions = 4   # matches Breakout default; agent_train reads this
    class app:
        @staticmethod
        def step(): pass
    def reset(self):            return torch.zeros(1, 6, 64, 64)
    def get_observation(self, **kw): return torch.zeros(1, 6, 64, 64)
    def step(self, action, **kw):
        return torch.rand(1, 6, 64, 64), float(np.random.randn() * 0.01), False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_steps=200_000, n=128, save_every=2_000, use_real_world=True,
          checkpoint_prefix: str = 'agent', env_id: str = 'ALE/Breakout-v5'):
    """
    Online n-step A2C with discrete Categorical policy for Atari ALE.

    Collect n steps with no gradient (O(1) forward_step per frame).
    Compute n-step returns backwards from a bootstrap value.
    Replay with gradients, detaching h after every 4 frames (TBPTT-4).
    One optimizer step.  Repeat.

    h is never zeroed between updates or between lives.
    The wave accumulates the full run history into its slow layers permanently.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # World first — n_actions is an env property, not a hyperparameter
    if use_real_world == 'gym':
        from gym_world import GymWorld
        world = GymWorld(env_id=env_id)
    elif use_real_world:
        from gameworld import GameWorld
        world = GameWorld()
        print("Using real Ursina gameworld")
    else:
        world = DummyWorld()
        print("Using dummy world")

    agent     = SchrodingerAgent(dim=256, depth=6, n_actions=world.n_actions).to(device)
    optimizer = optim.AdamW(agent.trainable_params(), lr=3e-4,
                            eps=1e-5, weight_decay=0.01)
    base_lr = 3e-4
    gamma   = 0.99

    # Checkpoint resume — pick up where we left off
    import glob as _glob
    ckpts = sorted(_glob.glob(f'{checkpoint_prefix}_step*.pt'))
    start_step = 0
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device)
        agent.load_state_dict(ckpt['agent'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0)
        print(f'Resumed from {ckpts[-1]} (step {start_step})')
    else:
        print('No checkpoint found — starting fresh.')

    n_params = sum(p.numel() for p in agent.parameters()) / 1e6
    print(f"Agent: {n_params:.2f}M params  |  n={n}  |  depth=6\n")

    obs = world.reset()
    if use_real_world == True:   # Ursina needs a second pump to get a real screenshot
        obs = world.get_observation(save_debug=True)
    elif use_real_world == 'gym':
        world.get_observation(save_debug=True)  # just saves debug_obs.png
    h = agent.init_hidden(device)

    rewards_window = deque(maxlen=200)
    step = start_step
    update = 0
    log_r = log_loss = log_g = log_ent = 0.0

    while step < n_steps:
        ent_coef = max(0.0001, 0.0003 * (1.0 - step / n_steps))

        # ── collect ───────────────────────────────────────────────────────
        obs_buf, act_buf, rew_buf, done_buf, old_lp_buf = [], [], [], [], []
        h_start = [t.detach().clone() for t in h]

        for _ in range(n):
            obs_t = obs.to(device)
            with torch.no_grad():
                logits, _, h_next = agent(obs_t, h)
                # NaN guard: if a previous bad update corrupted h, reinitialise rather than crash.
                # This can only trigger if tau blowup or cascading z amplification produced NaN
                # in the parameters — the z.clamp and tau softplus fixes above prevent recurrence.
                if torch.isnan(logits).any():
                    print("  !! NaN logits — reinitialising h")
                    h = agent.init_hidden(device)
                    logits, _, h_next = agent(obs_t, h)
                # Categorical sample — [B] LongTensor, values in [0, n_actions)
                dist_old = torch.distributions.Categorical(logits=logits)
                action   = dist_old.sample()
                old_lp_buf.append(dist_old.log_prob(action).detach())
            if use_real_world:
                world.app.step()
            obs, reward, done = world.step(action.squeeze(0))
            obs_buf.append(obs_t)
            act_buf.append(action)
            rew_buf.append(float(reward))
            done_buf.append(done)
            h = [t.detach() for t in h_next]
            step += 1

        # ── n-step returns ────────────────────────────────────────────────
        with torch.no_grad():
            _, nv, _ = agent(obs.to(device), h)
        R = nv.squeeze().detach()
        returns = []
        for r, d in zip(reversed(rew_buf), reversed(done_buf)):
            R = r + gamma * R * (1.0 - float(d))
            returns.insert(0, R)
        returns_t = torch.stack(returns).to(device)

        # ── normalise returns: O(1) scale for both value loss and policy ──
        ret_mean  = returns_t.mean().detach()
        ret_std   = returns_t.std().detach() + 1e-8
        returns_t = (returns_t - ret_mean) / ret_std

        # ── advantage normalisation: no-grad pass to collect values ──────
        with torch.no_grad():
            h_tmp = [t.clone() for t in h_start]
            vals  = []
            for i in range(n):
                _, v, h_tmp = agent(obs_buf[i], h_tmp)
                vals.append(v.squeeze())
                h_tmp = [t.detach() for t in h_tmp]
        adv_t = returns_t - torch.stack(vals)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Clear no-grad cache so _A() recomputes with gradients during replay.
        # Without this, dt/log_gamma/omega receive zero gradient every update.
        agent.transformer.invalidate_A_cache()

        # ── replay with gradients, TBPTT-8 ───────────────────────────────
        # Detach h every 8 frames (128 tokens) — full rollout = one gradient unit.
        # Gives dt 2× more gradient path vs TBPTT-4 to accelerate timescale split.
        h_r        = h_start
        total_loss = torch.tensor(0.0, device=device)
        ep_ent     = 0.0
        world_coef = 0.01  # world model loss weight — auxiliary only; 0.1 competed with critic gradient in L5
        for i in range(n):
            h_r_prev = h_r                                       # belief BEFORE obs[i]
            logits, val, h_r = agent(obs_buf[i], h_r)
            dist     = torch.distributions.Categorical(logits=logits)
            # act_buf[i] is [B] LongTensor — Categorical.log_prob needs no .sum()
            log_prob = dist.log_prob(act_buf[i].squeeze(-1))   # [B]
            entropy  = dist.entropy()                          # [B]
            # PPO clipped surrogate — bounds p_new/p_old to [0.8, 1.2].
            # Critical here: replay h diverges from collect h (wave is mutable),
            # so ratio can be large without clipping → causes entropy oscillation.
            ratio   = torch.exp(log_prob - old_lp_buf[i])     # [B] importance weight
            pg_loss = -torch.min(
                ratio * adv_t[i],
                torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_t[i]
            )
            # World model: h_prev should predict h_next (Koopman operator).
            # h_r[:,-1,1,:].real = deepest True wave's real part after obs[i].
            # Detach target: we supervise the predictor, not the dynamics.
            world_loss = F.mse_loss(
                agent.patch_pred(h_r_prev[-1][:, 1, :].real),
                h_r[-1][:, 1, :].real.detach()
            )
            total_loss = (total_loss
                          + pg_loss
                          + 0.5 * F.mse_loss(val.squeeze(), returns_t[i])
                          - ent_coef * entropy
                          + world_coef * world_loss)
            ep_ent += entropy.mean().item()
            if (i + 1) % 8 == 0:
                h_r = [t.detach() for t in h_r]   # TBPTT-8: gradient unit = 8 frames × 16 tokens

        total_loss = total_loss / n
        optimizer.zero_grad()
        total_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        raw_gnorm = gnorm.item()

        if not np.isfinite(raw_gnorm):
            # NaN/Inf gradients — applying would corrupt parameters.
            # Discard this update; the z.clamp + tau fixes prevent recurrence.
            optimizer.zero_grad()
            print(f"  !! Skipped update (gnorm={raw_gnorm})")
        else:
            # Reactive LR: same formula as the language model training loop
            stability   = 1.0 / (raw_gnorm + 1e-6)
            reactive_lr = base_lr / (1.0 + np.exp(-(stability - 0.5)))
            reactive_lr = float(np.clip(reactive_lr, base_lr * 0.1, base_lr * 2.0))
            for pg in optimizer.param_groups:
                pg['lr'] = reactive_lr
            optimizer.step()
        # Cache is already invalid (cleared before replay); stays invalid until
        # next collect phase calls _A(), at which point it repopulates correctly.

        update   += 1
        log_r    += sum(rew_buf)
        log_loss += total_loss.item()
        log_g    += raw_gnorm if np.isfinite(raw_gnorm) else 0.0
        log_ent  += ep_ent / n

        rewards_window.append(log_r)
        mean_r = np.mean(rewards_window) if rewards_window else 0.0
        dts = " ".join(f"{op.dt.item():.4f}" for op in agent.transformer.operators)
        aw  = F.softmax(agent.actor_layer_w,  dim=0).detach().cpu().numpy()
        cw  = F.softmax(agent.critic_layer_w, dim=0).detach().cpu().numpy()
        _lr_display = reactive_lr if np.isfinite(raw_gnorm) else 0.0
        print(
            f"[{update:5d}] step={step:7d} score={world.score} | "
            f"R={mean_r:+.4f} rr={log_r:+.4f} | "
            f"loss={log_loss:+.4f} gnorm={log_g:.3f} ent={log_ent:.2f} lr={_lr_display:.2e}\n"
            f"         dt=[{dts}]\n"
            f"         actor=[{' '.join(f'{w:.3f}' for w in aw)}]"
            f"  critic=[{' '.join(f'{w:.3f}' for w in cw)}]"
        )
        log_r = log_loss = log_g = log_ent = 0.0

        if step % save_every < n:
            path = f"{checkpoint_prefix}_step{step:07d}.pt"
            torch.save({'step': step, 'agent': agent.state_dict(),
                        'optimizer': optimizer.state_dict()}, path)
            print(f"  Saved {path}")

    print("\nTraining complete.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='WSSM Atari Agent')
    p.add_argument('--dummy',  action='store_true', help='Use dummy world (no gym needed)')
    p.add_argument('--ursina', action='store_true', help='Use Ursina 3D gameworld')
    p.add_argument('--env',    default='ALE/Seaquest-v5',
                   help='Atari env id (default: ALE/Seaquest-v5)\n'
                        'Options: ALE/Breakout-v5  ALE/Pong-v5  ALE/SpaceInvaders-v5\n'
                        '         ALE/Seaquest-v5  ALE/MontezumaRevenge-v5')
    p.add_argument('--prefix', default='agent_seaquest',
                   help='Checkpoint filename prefix (default: agent_seaquest).\n'
                        'Breakout used "agent".  Change when switching envs to avoid\n'
                        'loading mismatched policy heads.')
    p.add_argument('--steps',  type=int, default=500_000,
                   help='Total environment steps (default: 500k for Seaquest)')
    p.add_argument('--n',      type=int, default=256,
                   help='n-step rollout length (default: 256).\n'
                        'Seaquest oxygen cycle ~200-300 frames — n=256 captures\n'
                        'one full dive-and-surface in a single rollout.')
    args = p.parse_args()

    if args.dummy:
        train(use_real_world=False, n_steps=args.steps, n=args.n)
    elif args.ursina:
        train(use_real_world=True,  n_steps=args.steps, n=args.n)
    else:
        train(use_real_world='gym', n_steps=args.steps, n=args.n,
              env_id=args.env, checkpoint_prefix=args.prefix)
