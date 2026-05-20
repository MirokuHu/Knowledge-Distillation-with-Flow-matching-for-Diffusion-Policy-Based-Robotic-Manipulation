if __name__ == "__main__":
    import sys
    import os
    import pathlib

    THIS_FILE = pathlib.Path(__file__).resolve()

    # Find project root: the outer diffusion_policy repo directory.
    # It should contain eval.py and the diffusion_policy package folder.
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

import os
import json
import click
import torch
from torch.utils.data import DataLoader, random_split

from teacher_dataset import TeacherDataset
from policy import build_policy


@click.command()
@click.option("--model_type", default="mlp",
              type=click.Choice(["mlp", "rnn", "transformer"]))
@click.option("--data_path", required=True)
@click.option("--save_path", required=True)
@click.option("--device", default="cuda:0")
@click.option("--batch_size", default=256)
@click.option("--epochs", default=30)
@click.option("--lr", default=1e-4)
@click.option("--action_key", default="teacher_action",
              type=click.Choice(["teacher_action", "dataset_action"]))
@click.option("--hidden_dim", default=256)
@click.option("--num_layers", default=2)
@click.option("--dropout", default=0.0)
@click.option("--d_model", default=128)
@click.option("--nhead", default=4)
@click.option("--dim_feedforward", default=512)
@click.option("--pooling", default="last", type=click.Choice(["last", "mean"]))
@click.option("--val_ratio", default=0.1)
@click.option("--seed", default=42)
def main(
    model_type,
    data_path,
    save_path,
    device,
    batch_size,
    epochs,
    lr,
    action_key,
    hidden_dim,
    num_layers,
    dropout,
    d_model,
    nhead,
    dim_feedforward,
    pooling,
    val_ratio,
    seed,
):
    device = torch.device(device)

    print("========== Student BC Training ==========")
    print("model_type:", model_type)
    print("data_path:", data_path)
    print("save_path:", save_path)
    print("device:", device)
    print("action_key:", action_key)
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("lr:", lr)

    dataset = TeacherDataset(data_path, action_key=action_key)

    sample = dataset[0]
    obs_steps, obs_dim = sample["obs"].shape
    action_steps, action_dim = sample["action"].shape

    print("\n========== Auto-detected Shape ==========")
    print("obs shape:", tuple(sample["obs"].shape))
    print("action shape:", tuple(sample["action"].shape))
    print("obs_steps:", obs_steps)
    print("obs_dim:", obs_dim)
    print("action_steps:", action_steps)
    print("action_dim:", action_dim)
    print("dataset length:", len(dataset))

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

    if model_type == "transformer":
        model_kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "pooling": pooling,
        }
    else:
        model_kwargs = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    model = build_policy(
        model_type=model_type,
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim,
        model_kwargs=model_kwargs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n========== Model ==========")
    print(model)
    print("trainable parameters:", n_params)
    print("model_kwargs:", json.dumps(model_kwargs, indent=2))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            obs = batch["obs"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)

            pred = model(obs)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * obs.shape[0]
            train_count += obs.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device, non_blocking=True)
                target = batch["action"].to(device, non_blocking=True)

                pred = model(obs)
                loss = torch.nn.functional.mse_loss(pred, target)

                val_loss_sum += loss.item() * obs.shape[0]
                val_count += obs.shape[0]

        val_loss = val_loss_sum / max(1, val_count)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.6f} | "
            f"val_loss {val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            ckpt = {
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "obs_steps": obs_steps,
                "obs_dim": obs_dim,
                "action_steps": action_steps,
                "action_dim": action_dim,
                "action_key": action_key,
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