import torch
import torch.nn as nn


class ReactionModel(nn.Module):
    def __init__(self, emb_dim: int, n_users: int, hidden_dim: int = 128):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, 32)

        self.net = nn.Sequential(
            nn.Linear(emb_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_idx, item_emb):
        u = self.user_emb(user_idx)
        x = torch.cat([u, item_emb], dim=1)
        return self.net(x).squeeze(1)