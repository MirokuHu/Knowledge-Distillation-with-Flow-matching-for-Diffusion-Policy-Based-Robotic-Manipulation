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
import dill
import hydra
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
import click


@click.command()
@click.option('--checkpoint', required=True)
@click.option('--output', required=True)
@click.option('--device', default='cuda:0')
def main(checkpoint, output, device):
    device = torch.device(device)

    # ===== load checkpoint =====
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    print("workspace target:", cfg._target_)
    print("dataset target:", cfg.task.dataset._target_)
    if 'policy' in cfg:
        print("policy target:", cfg.policy._target_)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)

    # ===== get teacher =====
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()

    print("policy class:", policy.__class__.__name__)
    print("policy n_obs_steps:", getattr(policy, "n_obs_steps", "N/A"))
    print("policy n_action_steps:", getattr(policy, "n_action_steps", "N/A"))

    # ===== load dataset =====
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = dict_apply(batch, lambda x: x.to(device))

            # ===== lowdim obs =====
            obs = batch['obs'][:, :policy.n_obs_steps]   # [B, 2, 20]

            print("batch id", batch_idx)
            obs_dict = {
                'obs': obs
            }
            out = policy.predict_action(obs_dict)

            action = out['action']   # [B, 8, 2]
            gt_action = batch["action"][:, :policy.n_action_steps]  # [B, 8, 2]

            if batch_idx == 0:
                print("obs.shape:", obs.shape)
                print("action(gt).shape:", batch["action"].shape)
                print("predict_action keys:", list(out.keys()))
                for k, v in out.items():
                    print(f"{k}: shape={getattr(v, 'shape', None)}")

            for i in range(action.shape[0]):
                samples.append({
                    'obs': obs[i].cpu(),
                    'teacher_action': action[i].cpu(),
                    'dataset_action': gt_action[i].cpu()
                })

    # 确认没问题后再取消 break 跑全量
    torch.save(samples, output)
    print(f"Saved {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()