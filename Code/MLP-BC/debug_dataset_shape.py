if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import torch
import dill
import hydra
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply


@click.command()
@click.option('--checkpoint', required=True)
@click.option('--device', default='cuda:0')
def main(checkpoint, device):
    device = torch.device(device)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    print("========== CONFIG ==========")
    print("workspace target:", cfg._target_)
    print("dataset target:", cfg.task.dataset._target_)
    print("env_runner target:", cfg.task.env_runner._target_)
    if "policy" in cfg:
        print("policy target:", cfg.policy._target_)
    print("use_ema:", cfg.training.use_ema)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()

    print("\n========== POLICY ==========")
    print("policy class:", policy.__class__.__name__)
    print("n_obs_steps:", getattr(policy, "n_obs_steps", "N/A"))
    print("n_action_steps:", getattr(policy, "n_action_steps", "N/A"))
    print("horizon:", getattr(policy, "horizon", "N/A"))
    print("obs_dim:", getattr(policy, "obs_dim", "N/A"))
    print("action_dim:", getattr(policy, "action_dim", "N/A"))

    print("\n========== DATASET ==========")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print("dataset class:", dataset.__class__.__name__)
    print("dataset length:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    print("\n========== BATCH BEFORE DEVICE ==========")
    print("batch type:", type(batch))
    if isinstance(batch, dict):
        print("batch keys:", list(batch.keys()))
        for k, v in batch.items():
            print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
            if isinstance(v, dict):
                print(f"  nested keys of {k}:", list(v.keys()))
                for kk, vv in v.items():
                    print(f"    {kk}: type={type(vv)}, shape={getattr(vv, 'shape', None)}")

    batch = dict_apply(batch, lambda x: x.to(device))

    print("\n========== TEACHER INPUT / OUTPUT ==========")
    obs = batch["obs"][:, :policy.n_obs_steps]
    obs_dict = {"obs": obs}

    print("obs used for policy:", obs.shape)
    print("dataset action full:", batch["action"].shape)
    print("dataset action used:", batch["action"][:, :policy.n_action_steps].shape)

    with torch.no_grad():
        out = policy.predict_action(obs_dict)

    print("predict_action keys:", list(out.keys()))
    for k, v in out.items():
        print(f"{k}: shape={getattr(v, 'shape', None)}")

    print("\n========== SUMMARY FOR BC ==========")
    print("BC obs shape:", obs[0].shape)
    print("BC teacher_action shape:", out["action"][0].shape)
    print("BC dataset_action shape:", batch["action"][0, :policy.n_action_steps].shape)


if __name__ == "__main__":
    main()