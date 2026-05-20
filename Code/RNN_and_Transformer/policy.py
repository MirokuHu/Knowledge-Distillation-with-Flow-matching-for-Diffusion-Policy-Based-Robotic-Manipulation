import torch
import torch.nn as nn


class MLPBCPolicy(nn.Module):
    """
    Feed-forward behavior cloning student.

    Input:
        obs: [B, obs_steps, obs_dim]
    Output:
        action: [B, action_steps, action_dim]
    """
    def __init__(
        self,
        obs_steps: int,
        obs_dim: int,
        action_steps: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_steps = action_steps
        self.action_dim = action_dim

        in_dim = obs_steps * obs_dim
        out_dim = action_steps * action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.reshape(obs.shape[0], -1)
        action = self.net(x)
        return action.reshape(obs.shape[0], self.action_steps, self.action_dim)


class RNNBCPolicy(nn.Module):
    """
    GRU-based recurrent behavior cloning student.

    Structure:
        obs sequence -> Linear embedding -> GRU encoder -> last hidden -> MLP head -> action chunk
    """
    def __init__(
        self,
        obs_steps: int,
        obs_dim: int,
        action_steps: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_steps * action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)          # [B, To, H]
        y, _ = self.rnn(x)                # [B, To, H]
        feat = y[:, -1]                   # [B, H]
        action = self.head(feat)
        return action.reshape(obs.shape[0], self.action_steps, self.action_dim)


class TransformerBCPolicy(nn.Module):
    """
    Transformer encoder based behavior cloning student.

    Structure:
        obs sequence -> token embedding -> positional embedding
        -> TransformerEncoder -> pooling -> MLP head -> action chunk
    """
    def __init__(
        self,
        obs_steps: int,
        obs_dim: int,
        action_steps: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        if pooling not in ["last", "mean"]:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        self.obs_steps = obs_steps
        self.obs_dim = obs_dim
        self.action_steps = action_steps
        self.action_dim = action_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.pooling = pooling

        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, obs_steps, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, action_steps * action_dim),
        )

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs) + self.pos_embed[:, :obs.shape[1], :]
        y = self.encoder(x)

        if self.pooling == "mean":
            feat = y.mean(dim=1)
        else:
            feat = y[:, -1]

        action = self.head(feat)
        return action.reshape(obs.shape[0], self.action_steps, self.action_dim)


def build_policy(
    model_type: str,
    obs_steps: int,
    obs_dim: int,
    action_steps: int,
    action_dim: int,
    model_kwargs: dict = None,
) -> nn.Module:
    model_kwargs = model_kwargs or {}

    if model_type == "mlp":
        return MLPBCPolicy(
            obs_steps=obs_steps,
            obs_dim=obs_dim,
            action_steps=action_steps,
            action_dim=action_dim,
            hidden_dim=model_kwargs.get("hidden_dim", 256),
        )

    if model_type == "rnn":
        return RNNBCPolicy(
            obs_steps=obs_steps,
            obs_dim=obs_dim,
            action_steps=action_steps,
            action_dim=action_dim,
            hidden_dim=model_kwargs.get("hidden_dim", 256),
            num_layers=model_kwargs.get("num_layers", 2),
            dropout=model_kwargs.get("dropout", 0.0),
        )

    if model_type == "transformer":
        return TransformerBCPolicy(
            obs_steps=obs_steps,
            obs_dim=obs_dim,
            action_steps=action_steps,
            action_dim=action_dim,
            d_model=model_kwargs.get("d_model", 128),
            nhead=model_kwargs.get("nhead", 4),
            num_layers=model_kwargs.get("num_layers", 2),
            dim_feedforward=model_kwargs.get("dim_feedforward", 512),
            dropout=model_kwargs.get("dropout", 0.1),
            pooling=model_kwargs.get("pooling", "last"),
        )

    raise ValueError(f"Unknown model_type: {model_type}")