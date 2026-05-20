if __name__ == "__main__":
    import sys
    import os
    import pathlib

    THIS_DIR = pathlib.Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR
    while PROJECT_ROOT.name != "diffusion_policy" and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIS_DIR))
    os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_steps=2, obs_dim=20, action_steps=8, action_dim=2):
        super().__init__()
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_steps = action_steps
        self.action_dim = action_dim

        in_dim = obs_steps * obs_dim
        out_dim = action_steps * action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, obs):
        # obs: [B, obs_steps, obs_dim]
        x = obs.reshape(obs.shape[0], -1)
        action = self.net(x)
        return action.reshape(obs.shape[0], self.action_steps, self.action_dim)