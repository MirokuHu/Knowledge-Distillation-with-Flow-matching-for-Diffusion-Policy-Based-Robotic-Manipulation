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

# use line-buffering for both stdout and stderr
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import json
import click
import hydra
import torch
import dill
import wandb
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from bc_policy import BCPolicy


class BCPolicyWrapper:
    def __init__(self, model):
        self.model = model
        self.inference_time_total = 0.0
        self.inference_calls = 0

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def reset(self):
        pass

    def predict_action(self, obs_dict):
        obs = obs_dict['obs'].to(self.device)

        use_cuda_timing = (
            torch.cuda.is_available()
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
        )

        if use_cuda_timing:
            torch.cuda.synchronize(self.device)

        t0 = time.perf_counter()

        with torch.no_grad():
            action = self.model(obs)

        if use_cuda_timing:
            torch.cuda.synchronize(self.device)

        t1 = time.perf_counter()

        self.inference_time_total += (t1 - t0)
        self.inference_calls += 1

        return {
            'action': action
        }

    def get_time_log(self):
        mean_time = self.inference_time_total / max(1, self.inference_calls)
        return {
            "policy_inference_time_total_sec": self.inference_time_total,
            "policy_inference_time_mean_sec": mean_time,
            "policy_inference_calls": self.inference_calls
        }


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-m', '--bc_model', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, bc_model, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    device = torch.device(device)

    ckpt = torch.load(bc_model, map_location=device)

    # New checkpoint format with metadata
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        obs_steps = ckpt["obs_steps"]
        obs_dim = ckpt["obs_dim"]
        action_steps = ckpt["action_steps"]
        action_dim = ckpt["action_dim"]

        print("========== Loaded BC metadata ==========")
        print("obs_steps:", obs_steps)
        print("obs_dim:", obs_dim)
        print("action_steps:", action_steps)
        print("action_dim:", action_dim)
        print("action_key:", ckpt.get("action_key", "N/A"))
        print("best_val_loss:", ckpt.get("best_val_loss", "N/A"))

        bc = BCPolicy(
            obs_steps=obs_steps,
            obs_dim=obs_dim,
            action_steps=action_steps,
            action_dim=action_dim
        )
        bc.load_state_dict(ckpt["model_state_dict"])

    # Backward compatibility for old Push-T-only state_dict
    else:
        print("WARNING: old BC checkpoint format detected.")
        print("Using default Push-T shape: obs=[2,20], action=[8,2].")
        bc = BCPolicy()
        bc.load_state_dict(ckpt)

    bc.to(device)
    bc.eval()

    policy = BCPolicyWrapper(bc).to(device).eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    eval_start_time = time.perf_counter()
    runner_log = env_runner.run(policy)
    eval_end_time = time.perf_counter()

    runner_log["total_eval_time_sec"] = eval_end_time - eval_start_time
    runner_log.update(policy.get_time_log())

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    print(json.dumps(json_log, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()