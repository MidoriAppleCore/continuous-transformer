"""
Atari ALE wrapper — same interface as GameWorld.

Install:
    pip install gymnasium[atari] ale-py
    AutoROM --accept-license

Default: ALE/Breakout-v5  (4 actions, well-studied, fast to train)
Pass --env ALE/Pong-v5 etc. to agent_train.py to switch.

Recommended games for WSSM benchmarking (published DQN/PPO/R2D2 baselines):
  ALE/Breakout-v5          —  4 actions, canonical DQN benchmark
  ALE/Pong-v5              —  6 actions, easiest (~200k steps to solve)
  ALE/SpaceInvaders-v5     —  6 actions, spatial + temporal memory
  ALE/Seaquest-v5          — 18 actions, long-term oxygen management
  ALE/MontezumaRevenge-v5  — 18 actions, hard exploration, ideal for WSSM memory test

Published baseline scores for comparison:
  DQN (Mnih 2015):   Breakout=401,  Pong=20.9,  SpaceInvaders=1976
  Rainbow:           Breakout=417,  Pong=20.9,  SpaceInvaders=18789
  PPO:               Breakout=274,  Pong=20.7
  R2D2 (LSTM):       The direct apples-to-apples comparison — recurrent policy.
                     WSSM replaces the LSTM hidden state with the wave field h.

Preprocessing follows Mnih et al. 2015:
  - RGB 210x160 → 64×64  (keeps colour; VisionEncoder handles RGB)
  - frameskip=4  (standard ALE)
  - Reward clipped to [-1, 1]
  - Episode done on life loss (training stability)
  - h is NOT reset on life loss — wave persists across lives intentionally
  - No frame stacking — WSSM wave state h replaces the frame stack entirely
  - FIRE reset for games that require it
"""

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)   # gymnasium 1.x requires explicit ALE registration
import numpy as np
import torch
import torch.nn.functional as F

# Games that need FIRE (action=1) after reset to start the round
_FIRE_ENVS = {
    'ALE/Breakout-v5', 'ALE/SpaceInvaders-v5', 'ALE/Pong-v5',
    'ALE/Assault-v5',  'ALE/Phoenix-v5',        'ALE/DemonAttack-v5',
}


class GymWorld:
    """
    Atari ALE with the GameWorld / CarRacing interface.

    Attributes
    ----------
    n_actions : int — read by agent_train to size the policy head
    score     : int — game-over count (not life-loss count)
    """

    class app:
        """No-op frame pump — agent_train calls world.app.step() for Ursina."""
        @staticmethod
        def step():
            pass

    def __init__(self, env_id: str = 'ALE/Breakout-v5'):
        self.env_id    = env_id
        self.env       = gym.make(env_id, render_mode='rgb_array',
                                  frameskip=4, repeat_action_probability=0.0)
        self.n_actions = self.env.action_space.n
        self.score     = 0
        self._lives    = 0
        self._obs      = None
        self._fire      = env_id in _FIRE_ENVS
        self._frame_buf = None   # [1, 6, 64, 64] rolling 2-frame buffer; set in reset()

        raw, info   = self.env.reset()
        self._lives = info.get('lives', 0)
        self._obs   = raw
        print(f"GymWorld: {env_id}  raw_obs={raw.shape}  n_actions={self.n_actions}")
        print(f"  fire_reset={self._fire}  lives={self._lives}")

    # ------------------------------------------------------------------
    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 numpy  →  [1, 3, 64, 64] uint8 tensor.
        All-PyTorch path: no PIL, no numpy round-trip.
        Resize on CPU intentionally — 64×64 × 3B = 12 KB over PCIe vs 101 KB for full frame.
        """
        t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        t = F.interpolate(t, size=(64, 64), mode='bilinear', align_corners=False)
        return t.byte()   # back to uint8 — VisionEncoder reads img.max() > 1.5 to detect this

    def _fire_reset(self) -> None:
        """Press FIRE (action=1) to start the round — some games require it."""
        if not self._fire:
            return
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset()
        self._obs   = obs
        self._lives = info.get('lives', self._lives)
        # Sync frame buffer after FIRE — duplicate so t-1 and t are both fresh
        t = self._tensor(self._obs)
        self._frame_buf = torch.cat([t, t], dim=1)

    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        raw, info   = self.env.reset()
        self._lives = info.get('lives', 0)
        self._obs   = raw
        t = self._tensor(self._obs)
        self._frame_buf = torch.cat([t, t], dim=1)   # [1, 6, 64, 64] — t-1=t=first frame
        self._fire_reset()   # updates _frame_buf to post-FIRE frame
        return self._frame_buf

    def get_observation(self, save_debug: bool = False) -> torch.Tensor:
        if save_debug and self._obs is not None:
            t = self._tensor(self._obs)
            import torchvision
            torchvision.utils.save_image(t.float() / 255.0, 'debug_obs.png')
            print("Saved debug_obs.png")
        return self._frame_buf

    # ------------------------------------------------------------------
    def step(self, action, **kwargs):
        """
        action : int scalar or [1] LongTensor — discrete action index
        returns: obs [1,3,64,64] uint8 tensor, reward float, done bool
        """
        a = int(action.item()) if hasattr(action, 'item') else int(action)
        a = a % self.n_actions   # safety clamp

        raw, reward, terminated, truncated, info = self.env.step(a)
        self._obs = raw

        # Life loss = done for return computation (training stability standard).
        # h is intentionally NOT reset — wave persists across lives.
        lives     = info.get('lives', self._lives)
        life_lost = lives < self._lives
        self._lives = lives
        done = terminated or truncated or life_lost

        if terminated or truncated:
            self.score += 1
            raw, info   = self.env.reset()
            self._obs   = raw
            self._lives = info.get('lives', 0)
            t = self._tensor(self._obs)
            self._frame_buf = torch.cat([t, t], dim=1)  # safe for non-FIRE games
            self._fire_reset()   # overwrites _frame_buf if FIRE is needed
        else:
            # Roll: drop oldest 3 channels, append current frame
            t = self._tensor(self._obs)
            self._frame_buf = torch.cat([self._frame_buf[:, 3:], t], dim=1)  # [1,6,64,64]

        # Clip to [-1, 1] — prevents value scale explosion across games (Mnih 2015)
        reward = float(np.clip(reward, -1.0, 1.0))
        return self._frame_buf, reward, done
