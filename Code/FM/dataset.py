import os
import torch
from torch.utils.data import Dataset


class FlowMatchingDataset(Dataset):
    """
    Each item in teacher_samples.pt should contain:
        obs: [obs_steps, obs_dim]
        teacher_action: [action_steps, action_dim]
        dataset_action: [action_steps, action_dim]
    """
    def __init__(self, path: str, action_key: str = "teacher_action"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Teacher sample file not found: {path}")

        self.data = torch.load(path)
        self.action_key = action_key

        if len(self.data) == 0:
            raise RuntimeError(f"Empty teacher sample file: {path}")

        keys = list(self.data[0].keys())
        if "obs" not in keys:
            raise KeyError(f"'obs' not found. Available keys: {keys}")
        if action_key not in keys:
            raise KeyError(f"action_key='{action_key}' not found. Available keys: {keys}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "obs": item["obs"].float(),
            "target_action": item[self.action_key].float(),
        }