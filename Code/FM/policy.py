import math
import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] or [B, 1]
    return: [B, dim]
    """
    if t.dim() == 2:
        t = t.squeeze(-1)

    half = dim // 2
    device = t.device

    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )

    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


class FlowMLPPolicy(nn.Module):
    """
    Conditional Flow Matching student.

    Inputs:
        obs: [B, obs_steps, obs_dim]
        x_t: [B, action_steps, action_dim]
        t:   [B] or [B, 1] or [B, 1, 1]

    Output:
        velocity: [B, action_steps, action_dim]
    """
    def __init__(
        self,
        obs_steps: int,
        obs_dim: int,
        action_steps: int,
        action_dim: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        obs_flat_dim = obs_steps * obs_dim
        action_flat_dim = action_steps * action_dim
        input_dim = obs_flat_dim + action_flat_dim + time_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_flat_dim),
        )

    def forward(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]

        if t.dim() == 3:
            t = t.reshape(b)
        elif t.dim() == 2:
            t = t.squeeze(-1)

        obs_flat = obs.reshape(b, -1)
        x_flat = x_t.reshape(b, -1)
        t_emb = sinusoidal_time_embedding(t, self.time_embed_dim)

        inp = torch.cat([obs_flat, x_flat, t_emb], dim=-1)
        v = self.net(inp)

        return v.reshape(b, self.action_steps, self.action_dim)


def build_flow_policy(
    obs_steps: int,
    obs_dim: int,
    action_steps: int,
    action_dim: int,
    model_kwargs: dict = None,
) -> FlowMLPPolicy:
    model_kwargs = model_kwargs or {}
    return FlowMLPPolicy(
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim,
        hidden_dim=model_kwargs.get("hidden_dim", 512),
        time_embed_dim=model_kwargs.get("time_embed_dim", 64),
    )