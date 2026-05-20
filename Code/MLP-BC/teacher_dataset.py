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
from torch.utils.data import Dataset


class TeacherDataset(Dataset):
    def __init__(self, path, action_key="teacher_action"):
        self.data = torch.load(path)
        self.action_key = action_key

        if len(self.data) == 0:
            raise RuntimeError(f"Empty teacher dataset: {path}")

        if action_key not in self.data[0]:
            raise KeyError(
                f"action_key='{action_key}' not found. "
                f"Available keys: {list(self.data[0].keys())}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "obs": item["obs"].float(),
            "action": item[self.action_key].float()
        }