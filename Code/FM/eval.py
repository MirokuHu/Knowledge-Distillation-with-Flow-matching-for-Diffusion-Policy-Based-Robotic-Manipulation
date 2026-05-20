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
from policy import build_flow_policy
from distill.BC.policy import build_policy as build_bc_policy


def load_bc_prior_model(bc_model_path: str, device: torch.device):
    import importlib.util
    import pathlib

    bc_policy_path = pathlib.Path(__file__).resolve().parents[1] / "BC" / "policy.py"

    if not bc_policy_path.exists():
        raise FileNotFoundError(f"BC policy.py not found: {bc_policy_path}")

    spec = importlib.util.spec_from_file_location("bc_policy_module", bc_policy_path)
    bc_policy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc_policy_module)

    build_bc_policy = bc_policy_module.build_policy

    ckpt = torch.load(bc_model_path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Unsupported BC prior checkpoint format: {bc_model_path}. "
            "Please use checkpoint trained by distill/BC/train.py"
        )

    model = build_bc_policy(
        model_type=ckpt["model_type"],
        obs_steps=ckpt["obs_steps"],
        obs_dim=ckpt["obs_dim"],
        action_steps=ckpt["action_steps"],
        action_dim=ckpt["action_dim"],
        model_kwargs=ckpt.get("model_kwargs", {}),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, ckpt


class FlowPolicyWrapper:
    def __init__(
        self,
        flow_model,
        prior_type="standard",
        source_std=1.0,
        n_flow_steps=3,
        bc_prior=None,
    ):
        self.flow_model = flow_model
        self.prior_type = prior_type
        self.source_std = source_std
        self.n_flow_steps = n_flow_steps
        self.bc_prior = bc_prior

        self.inference_time_total = 0.0
        self.inference_calls = 0

    @property
    def dtype(self):
        return next(self.flow_model.parameters()).dtype

    @property
    def device(self):
        return next(self.flow_model.parameters()).device

    def to(self, device):
        self.flow_model.to(device)
        if self.bc_prior is not None:
            self.bc_prior.to(device)
        return self

    def eval(self):
        self.flow_model.eval()
        if self.bc_prior is not None:
            self.bc_prior.eval()
        return self

    def reset(self):
        pass

    def sample_source(self, obs):
        b = obs.shape[0]
        action_steps = self.flow_model.action_steps
        action_dim = self.flow_model.action_dim

        if self.prior_type == "standard":
            return torch.randn(
                b,
                action_steps,
                action_dim,
                device=self.device,
                dtype=obs.dtype,
            ) * self.source_std

        if self.prior_type == "bc":
            if self.bc_prior is None:
                raise RuntimeError("bc_prior is required when prior_type='bc'")
            with torch.no_grad():
                mean = self.bc_prior(obs)
            return mean + self.source_std * torch.randn_like(mean)

        raise ValueError(f"Unknown prior_type: {self.prior_type}")

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
            x = self.sample_source(obs)

            # Euler integration from source side t=1 to target side t=0.
            # Learned velocity points from source to target: v = target - source.
            dt = 1.0 / float(self.n_flow_steps)

            for i in range(self.n_flow_steps):
                t_value = 1.0 - float(i) / float(self.n_flow_steps)
                t = torch.full(
                    (obs.shape[0], 1, 1),
                    t_value,
                    device=self.device,
                    dtype=obs.dtype,
                )

                v = self.flow_model(obs, x, t)
                x = x + dt * v

            action = x

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
@click.option("-m", "--fm_model", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--n_flow_steps", default=3)
@click.option("--override_prior_type", default=None,
              type=click.Choice(["standard", "bc"]))
@click.option("--override_source_std", default=None, type=float)
@click.option("--override_bc_model", default=None)
def main(
    checkpoint,
    fm_model,
    output_dir,
    device,
    n_flow_steps,
    override_prior_type,
    override_source_std,
    override_bc_model,
):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load teacher checkpoint only for cfg / env_runner.
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    device = torch.device(device)

    fm_ckpt = torch.load(fm_model, map_location=device)

    if not isinstance(fm_ckpt, dict) or "model_state_dict" not in fm_ckpt:
        raise RuntimeError(
            "Unsupported FM checkpoint format. "
            "Please train with distill/FM/train.py"
        )

    obs_steps = fm_ckpt["obs_steps"]
    obs_dim = fm_ckpt["obs_dim"]
    action_steps = fm_ckpt["action_steps"]
    action_dim = fm_ckpt["action_dim"]
    model_kwargs = fm_ckpt.get("model_kwargs", {})

    prior_type = override_prior_type or fm_ckpt.get("prior_type", "standard")
    source_std = override_source_std if override_source_std is not None else fm_ckpt.get("source_std", 1.0)
    bc_model = override_bc_model or fm_ckpt.get("bc_model", None)

    print("========== Loaded FM Checkpoint ==========")
    print("fm_model:", fm_model)
    print("obs_steps:", obs_steps)
    print("obs_dim:", obs_dim)
    print("action_steps:", action_steps)
    print("action_dim:", action_dim)
    print("prior_type:", prior_type)
    print("source_std:", source_std)
    print("bc_model:", bc_model)
    print("n_flow_steps:", n_flow_steps)
    print("best_val_loss:", fm_ckpt.get("best_val_loss", "N/A"))
    print("model_kwargs:", json.dumps(model_kwargs, indent=2))

    flow_model = build_flow_policy(
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim,
        model_kwargs=model_kwargs,
    )

    flow_model.load_state_dict(fm_ckpt["model_state_dict"])
    flow_model.to(device)
    flow_model.eval()

    bc_prior = None
    bc_prior_ckpt = None

    if prior_type == "bc":
        if bc_model is None:
            raise RuntimeError("bc_model is required for BC prior evaluation.")
        bc_prior, bc_prior_ckpt = load_bc_prior_model(bc_model, device)

    policy = FlowPolicyWrapper(
        flow_model=flow_model,
        prior_type=prior_type,
        source_std=source_std,
        n_flow_steps=n_flow_steps,
        bc_prior=bc_prior,
    ).to(device).eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
    )

    eval_start_time = time.perf_counter()
    runner_log = env_runner.run(policy)
    eval_end_time = time.perf_counter()

    runner_log["total_eval_time_sec"] = eval_end_time - eval_start_time
    runner_log.update(policy.get_time_log())

    runner_log["student_model_type"] = "flow_matching"
    runner_log["fm_model_path"] = fm_model
    runner_log["fm_best_val_loss"] = fm_ckpt.get("best_val_loss", None)
    runner_log["fm_prior_type"] = prior_type
    runner_log["fm_source_std"] = source_std
    runner_log["fm_n_flow_steps"] = n_flow_steps
    runner_log["fm_action_key"] = fm_ckpt.get("action_key", None)
    runner_log["fm_n_params"] = fm_ckpt.get("n_params", None)
    runner_log["fm_bc_model"] = bc_model

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