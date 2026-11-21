import torch
import torch.nn as nn

from mamba_ssm import Mamba


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ConvBranch(nn.Module):
    """Same CNN branch as the Keras models (kernel 11, 'same' padding)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 11, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 24, 11, stride=2, padding=5),
            nn.BatchNorm1d(24), nn.ReLU(),
            nn.MaxPool1d(3, ceil_mode=True),
            nn.Conv1d(24, 32, 11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(5, ceil_mode=True),
        )

    def forward(self, x):            # x: (B, 1, 900)
        return self.net(x)           # (B, 32, 30)


class ECABlock(nn.Module):
    """Efficient channel attention on (B, T, C) features."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):                        # x: (B, T, C)
        w = x.mean(dim=1).unsqueeze(1)           # (B, 1, C)
        w = torch.sigmoid(self.conv(w))          # (B, 1, C)
        return x * w


class PositionalEncoding(nn.Module):
    """Learnable positional embedding, as in the Keras model."""

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.05)

    def forward(self, x):
        return x + self.pos_emb


class MambaLayer(nn.Module):
    """Residual Mamba layer: x + Dropout(Mamba(LayerNorm(x)))."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))


class DualBranchMamba(nn.Module):

    def __init__(self, seq_len=30, d_model=64, dropout=0.1):
        super().__init__()
        self.rri_branch  = ConvBranch()
        self.ampl_branch = ConvBranch()
        self.eca         = ECABlock(3)
        self.pos         = PositionalEncoding(seq_len, d_model)
        self.mamba1      = MambaLayer(d_model, dropout)
        self.mamba2      = MambaLayer(d_model, dropout)
        self.fc1         = nn.Linear(2 * d_model, 64)
        self.drop        = nn.Dropout(dropout)
        self.fc2         = nn.Linear(64, 2)

    def forward(self, x):                        # x: (B, 900, 2)
        rri  = self.rri_branch(x[:, :, 0:1].transpose(1, 2))
        ampl = self.ampl_branch(x[:, :, 1:2].transpose(1, 2))
        h = torch.cat([rri, ampl], dim=1).transpose(1, 2)       # (B, 30, 64)
        h = self.eca(h)
        h = self.pos(h)
        h = self.mamba1(h).contiguous()
        h = self.mamba2(h)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        pooled = self.drop(torch.relu(self.fc1(pooled)))
        return self.fc2(pooled)                                  # logits (B, 2)
