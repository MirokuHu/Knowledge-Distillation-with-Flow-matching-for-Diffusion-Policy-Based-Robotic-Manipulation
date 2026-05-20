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
    sys.path.insert(0, str(PROJECT_ROOT / "distill" / "BC"))
    sys.path.insert(0, str(THIS_FILE.parent))
    os.chdir(PROJECT_ROOT)

import os
import json
import click
import torch
from torch.utils.data import DataLoader, random_split

from dataset import FlowMatchingDataset
from policy import build_flow_policy

# Import BC policy builder for BC prior.
from policy import build_flow_policy as _unused_to_avoid_lint
from distill.BC.policy import build_policy as build_bc_policy


def load_bc_prior_model(bc_model_path: str, device: torch.device):
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


@click.command()
@click.option("--data_path", required=True)
@click.option("--save_path", required=True)
@click.option("--device", default="cuda:0")
@click.option("--epochs", default=50)
@click.option("--batch_size", default=256)
@click.option("--lr", default=1e-4)
@click.option("--action_key", default="teacher_action",
              type=click.Choice(["teacher_action", "dataset_action"]))
@click.option("--prior_type", default="standard",
              type=click.Choice(["standard", "bc"]))
@click.option("--bc_model", default=None, help="Required when prior_type=bc")
@click.option("--source_std", default=1.0, help="sigma for source sampling")
@click.option("--hidden_dim", default=512)
@click.option("--time_embed_dim", default=64)
@click.option("--val_ratio", default=0.1)
@click.option("--seed", default=42)
def main(
    data_path,
    save_path,
    device,
    epochs,
    batch_size,
    lr,
    action_key,
    prior_type,
    bc_model,
    source_std,
    hidden_dim,
    time_embed_dim,
    val_ratio,
    seed,
):
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("========== Flow Matching Training ==========")
    print("data_path:", data_path)
    print("save_path:", save_path)
    print("prior_type:", prior_type)
    print("bc_model:", bc_model)
    print("source_std:", source_std)
    print("action_key:", action_key)
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("lr:", lr)

    dataset = FlowMatchingDataset(data_path, action_key=action_key)

    sample = dataset[0]
    obs_steps, obs_dim = sample["obs"].shape
    action_steps, action_dim = sample["target_action"].shape

    print("\n========== Auto-detected Shape ==========")
    print("obs:", tuple(sample["obs"].shape))
    print("target_action:", tuple(sample["target_action"].shape))
    print("dataset length:", len(dataset))

    bc_prior = None
    bc_prior_ckpt = None

    if prior_type == "bc":
        if bc_model is None:
            raise ValueError("--bc_model is required when --prior_type bc")
        bc_prior, bc_prior_ckpt = load_bc_prior_model(bc_model, device)

        # Safety check.
        if (
            bc_prior_ckpt["obs_steps"] != obs_steps
            or bc_prior_ckpt["obs_dim"] != obs_dim
            or bc_prior_ckpt["action_steps"] != action_steps
            or bc_prior_ckpt["action_dim"] != action_dim
        ):
            raise RuntimeError(
                "BC prior shape does not match teacher samples.\n"
                f"BC: obs=[{bc_prior_ckpt['obs_steps']},{bc_prior_ckpt['obs_dim']}], "
                f"action=[{bc_prior_ckpt['action_steps']},{bc_prior_ckpt['action_dim']}]\n"
                f"Data: obs=[{obs_steps},{obs_dim}], action=[{action_steps},{action_dim}]"
            )

    model_kwargs = {
        "hidden_dim": hidden_dim,
        "time_embed_dim": time_embed_dim,
    }

    model = build_flow_policy(
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim,
        model_kwargs=model_kwargs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Flow Model ==========")
    print(model)
    print("trainable parameters:", n_params)
    print("model_kwargs:", json.dumps(model_kwargs, indent=2))

    n_total = len(dataset)
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def make_source(obs, target):
        if prior_type == "standard":
            return torch.randn_like(target) * source_std

        if prior_type == "bc":
            with torch.no_grad():
                mean = bc_prior(obs)
            return mean + source_std * torch.randn_like(mean)

        raise ValueError(prior_type)

    def fm_loss_batch(batch):
        obs = batch["obs"].to(device, non_blocking=True)
        target = batch["target_action"].to(device, non_blocking=True)

        source = make_source(obs, target)

        b = obs.shape[0]
        t = torch.rand(b, 1, 1, device=device)

        # t=0: target action
        # t=1: source sample
        x_t = (1.0 - t) * target + t * source
        v_target = target - source

        v_pred = model(obs, x_t, t)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return loss, obs.shape[0]

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            loss, count = fm_loss_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * count
            train_count += count

        train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                loss, count = fm_loss_batch(batch)
                val_loss_sum += loss.item() * count
                val_count += count

        val_loss = val_loss_sum / max(1, val_count)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            ckpt = {
                "model_type": "flow_mlp",
                "model_state_dict": model.state_dict(),
                "obs_steps": obs_steps,
                "obs_dim": obs_dim,
                "action_steps": action_steps,
                "action_dim": action_dim,
                "action_key": action_key,
                "prior_type": prior_type,
                "source_std": source_std,
                "bc_model": bc_model,
                "model_kwargs": model_kwargs,
                "best_val_loss": best_val,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_ratio": val_ratio,
                "seed": seed,
                "data_path": data_path,
                "n_params": n_params,
            }

            torch.save(ckpt, save_path)
            print(f"saved best model to {save_path}")

    print("\ntraining done.")
    print("best_val_loss:", best_val)


if __name__ == "__main__":
    main()