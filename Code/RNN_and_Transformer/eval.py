if __name__ == "__main__":
    import sys
    import os
    import pathlib

    THIS_FILE = pathlib.Path(__file__).resolve()

    PROJECT_ROOT = None
    for p in [THIS_FILE.parent] + list(THIS_FILE.parents):
        if (p / "eval.py").exists() and (p / "diffusion_policy").is_dir():
            PROJECT_ROOT = p
            break

    if PROJECT_ROOT is None:
        raise RuntimeError("Cannot find project root. Please run inside diffusion_policy repo.")

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIS_FILE.parent))
    os.chdir(PROJECT_ROOT)

# use line-buffering for both stdout and stderr
import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import json
import time
import pathlib
import click
import hydra
import torch
import dill
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from policy import build_policy


class StudentPolicyWrapper:
    """
    Wrap a BC student so that it has the same interface as Diffusion Policy:
        predict_action(obs_dict) -> {'action': action}
    Also records pure model inference time.
    """
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
        # Stateless student. Keep this method for env_runner compatibility.
        pass

    def predict_action(self, obs_dict):
        obs = obs_dict["obs"].to(self.device)

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

        return {"action": action}

    def get_time_log(self):
        mean_time = self.inference_time_total / max(1, self.inference_calls)
        return {
            "policy_inference_time_total_sec": self.inference_time_total,
            "policy_inference_time_mean_sec": mean_time,
            "policy_inference_calls": self.inference_calls,
        }


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-m", "--student_model", "--bc_model", "student_model", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(checkpoint, student_model, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load teacher checkpoint only for cfg and env_runner.
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    device = torch.device(device)

    student_ckpt = torch.load(student_model, map_location=device)

    if not isinstance(student_ckpt, dict) or "model_state_dict" not in student_ckpt:
        raise RuntimeError(
            "Unsupported student checkpoint format. "
            "Please train with distill/BC/train.py."
        )

    model_type = student_ckpt["model_type"]
    obs_steps = student_ckpt["obs_steps"]
    obs_dim = student_ckpt["obs_dim"]
    action_steps = student_ckpt["action_steps"]
    action_dim = student_ckpt["action_dim"]
    model_kwargs = student_ckpt.get("model_kwargs", {})

    print("========== Loaded Student Checkpoint ==========")
    print("student_model:", student_model)
    print("model_type:", model_type)
    print("obs_steps:", obs_steps)
    print("obs_dim:", obs_dim)
    print("action_steps:", action_steps)
    print("action_dim:", action_dim)
    print("action_key:", student_ckpt.get("action_key", "N/A"))
    print("best_val_loss:", student_ckpt.get("best_val_loss", "N/A"))
    print("model_kwargs:", json.dumps(model_kwargs, indent=2))

    model = build_policy(
        model_type=model_type,
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim,
        model_kwargs=model_kwargs,
    )

    model.load_state_dict(student_ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    policy = StudentPolicyWrapper(model).to(device).eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
    )

    eval_start_time = time.perf_counter()
    runner_log = env_runner.run(policy)
    eval_end_time = time.perf_counter()

    runner_log["total_eval_time_sec"] = eval_end_time - eval_start_time
    runner_log.update(policy.get_time_log())

    runner_log["student_model_type"] = model_type
    runner_log["student_model_path"] = student_model
    runner_log["student_best_val_loss"] = student_ckpt.get("best_val_loss", None)
    runner_log["student_action_key"] = student_ckpt.get("action_key", None)
    runner_log["student_n_params"] = student_ckpt.get("n_params", None)

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

    print(json.dumps(json_log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()