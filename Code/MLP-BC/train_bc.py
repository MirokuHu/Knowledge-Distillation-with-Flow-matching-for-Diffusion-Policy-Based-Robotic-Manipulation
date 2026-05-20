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

import os
import click
import torch
from torch.utils.data import DataLoader, random_split

from teacher_dataset import TeacherDataset
from bc_policy import BCPolicy


@click.command()
@click.option('--data_path', default='data/teacher_samples.pt')
@click.option('--save_path', default='data/bc_model.pt')
@click.option('--device', default='cuda:0')
@click.option('--batch_size', default=256)
@click.option('--epochs', default=30)
@click.option('--lr', default=1e-4)
@click.option('--action_key', default='teacher_action')
def main(data_path, save_path, device, batch_size, epochs, lr, action_key):
    device = torch.device(device)

    print("========== Train MLP-BC ==========")
    print("data_path:", data_path)
    print("save_path:", save_path)
    print("action_key:", action_key)
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("lr:", lr)

    dataset = TeacherDataset(data_path, action_key=action_key)

    sample = dataset[0]
    obs_steps, obs_dim = sample["obs"].shape
    action_steps, action_dim = sample["action"].shape

    print("========== Auto-detected shape ==========")
    print("obs shape:", sample["obs"].shape)
    print("action shape:", sample["action"].shape)
    print("obs_steps:", obs_steps)
    print("obs_dim:", obs_dim)
    print("action_steps:", action_steps)
    print("action_dim:", action_dim)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = BCPolicy(
        obs_steps=obs_steps,
        obs_dim=obs_dim,
        action_steps=action_steps,
        action_dim=action_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            target = batch["action"].to(device)

            pred = model(obs)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * obs.shape[0]
            train_count += obs.shape[0]

        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                target = batch["action"].to(device)

                pred = model(obs)
                loss = torch.nn.functional.mse_loss(pred, target)

                val_loss_sum += loss.item() * obs.shape[0]
                val_count += obs.shape[0]

        val_loss = val_loss_sum / val_count

        print(f"epoch {epoch:03d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_steps": obs_steps,
                "obs_dim": obs_dim,
                "action_steps": action_steps,
                "action_dim": action_dim,
                "action_key": action_key,
                "best_val_loss": best_val,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "data_path": data_path
            }, save_path)
            print(f"saved best model to {save_path}")

    print("training done.")
    print("best_val_loss:", best_val)


if __name__ == "__main__":
    main()