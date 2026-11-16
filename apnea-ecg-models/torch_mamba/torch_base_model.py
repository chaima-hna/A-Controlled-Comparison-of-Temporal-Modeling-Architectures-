# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
import torch
import torch.nn as nn

# =========================
# ECA BLOCK
# =========================

class ECABlock(nn.Module):
    """
    Efficient Channel Attention.
    Input/Output: (B, T, C)

    1. GAP over T  -> (B, C)
    2. Conv1d(k=3) -> (B, C)  learn inter-channel relationships
    3. Sigmoid     -> weights in [0, 1]
    4. Scale input channels
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad       = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)

    def forward(self, x):
        gap = x.mean(dim=1)              # (B, C)
        gap = gap.unsqueeze(1)           # (B, 1, C)
        w   = torch.sigmoid(self.conv(gap))  # (B, 1, C)
        return x * w                     # (B, T, C)


# =========================
# POSITIONAL ENCODING
# =========================

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Adds (1, seq_len, embed_dim) learned embedding to input.
    """

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(
            torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_emb


# =========================
# CONV BRANCH
# =========================

class ConvBranch(nn.Module):
    """
    Single CNN branch for one ECG signal (RRI or Amplitude).

    Input:  (B, 1, 900)
    Output: (B, 32, 30)

    900 -> stride=2 -> 450 -> MaxPool(3) -> 150 -> MaxPool(5) -> 30
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU())

        self.block2 = nn.Sequential(
            nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1))

        self.block3 = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5, padding=2))

    def forward(self, x):
        x = self.block1(x)   # (B, 16, 900)
        x = self.block2(x)   # (B, 24, 150)
        x = self.block3(x)   # (B, 32,  30)
        return x
