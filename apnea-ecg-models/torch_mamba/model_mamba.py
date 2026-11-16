# [split] import line(s) added so this file runs on its own; everything else is your original code, unchanged.
from torch_config import *
from torch_base_model import ECABlock, PositionalEncoding, ConvBranch

# =========================
# MAMBA WRAPPER BLOCK
# =========================

class MambaBlock(nn.Module):
    """
    Wrapper around the official mamba_ssm.Mamba class.
    Adds LayerNorm + residual connection (same pattern as TransformerBlock).

    The official Mamba class:
      Mamba(d_model, d_state, d_conv, expand)
      Input/Output: (B, L, d_model)

    Parameters (matching your model dimensions):
      d_model = 64   input/output dimension
      d_state = 16   SSM state size N
      d_conv  = 4    causal conv kernel size
      expand  = 2    d_inner = 64*2 = 128  (matches Transformer ff_dim)
    """

    def __init__(self, d_model=64, d_state=16, d_conv=4,
                 expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        # LayerNorm -> Mamba -> Dropout -> Residual
        return x + self.drop(self.mamba(self.norm(x)))


# =========================
# M7 — DUAL BRANCH MAMBA
# =========================

class DualBranchMamba(nn.Module):
    """
    M7 Full Model using official mamba-ssm.

    Input  (B, 900, 2)
    ├─ RRI  branch  -> (B, 32, 30)
    ├─ Ampl branch  -> (B, 32, 30)
    ├─ Concat       -> (B, 64, 30)
    ├─ Transpose    -> (B, 30, 64)
    ├─ ECA          -> (B, 30, 64)
    ├─ PosEnc       -> (B, 30, 64)
    ├─ MambaBlock   -> (B, 30, 64)
    ├─ MambaBlock   -> (B, 30, 64)
    ├─ GAP + GMP    -> (B, 128)
    ├─ Linear(64)   -> (B, 64)
    └─ Linear(2)    -> (B, 2)
    """

    def __init__(self, num_classes=2, dropout=DROPOUT):
        super().__init__()

        # Dual CNN branches
        self.rri_branch  = ConvBranch()
        self.ampl_branch = ConvBranch()

        # Channel attention + positional encoding
        self.eca    = ECABlock(channels=64, kernel_size=3)
        self.posenc = PositionalEncoding(seq_len=30, embed_dim=64)

        # Two Mamba blocks (official mamba-ssm)
        self.mamba1 = MambaBlock(d_model=64, d_state=16,
                                 d_conv=4, expand=2, dropout=dropout)
        self.mamba2 = MambaBlock(d_model=64, d_state=16,
                                 d_conv=4, expand=2, dropout=dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    def forward(self, x):
        # x: (B, 900, 2)

        rri  = x[:, :, 0].unsqueeze(1)    # (B, 1, 900)
        ampl = x[:, :, 1].unsqueeze(1)    # (B, 1, 900)

        rri_feat  = self.rri_branch(rri)   # (B, 32, 30)
        ampl_feat = self.ampl_branch(ampl) # (B, 32, 30)

        fused = torch.cat([rri_feat, ampl_feat], dim=1)  # (B, 64, 30)
        fused = fused.transpose(1, 2)                     # (B, 30, 64)

        fused = self.eca(fused)            # (B, 30, 64)
        fused = self.posenc(fused)         # (B, 30, 64)

        fused = self.mamba1(fused)         # (B, 30, 64)
        fused = self.mamba2(fused)         # (B, 30, 64)

        gap    = fused.mean(dim=1)         # (B, 64)
        gmp    = fused.max(dim=1).values   # (B, 64)
        pooled = torch.cat([gap, gmp], dim=1)  # (B, 128)

        return self.classifier(pooled)     # (B, 2)


# =========================
# PARAMETER COUNT
# =========================

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*62)
    print("  PARAMETER COUNT - M7 (Official Mamba)")
    print("="*62)
    print(f"  {'Module':<38} {'Params':>10}  {'%':>6}")
    print("  " + "-"*58)

    for name, module in model.named_children():
        n   = sum(p.numel() for p in module.parameters() if p.requires_grad)
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {name:<38} {n:>10,}  {pct:>5.1f}%")
        for sub_name, sub_module in module.named_children():
            n_sub = sum(p.numel() for p in sub_module.parameters()
                        if p.requires_grad)
            if n_sub > 0:
                pct_sub = 100.0 * n_sub / total
                label   = "  |-- " + sub_name
                print(f"  {label:<38} {n_sub:>10,}  {pct_sub:>5.1f}%")

    print("  " + "-"*58)
    print(f"  {'TOTAL TRAINABLE':<38} {total:>10,}  {'100.0%':>6}")
    print("="*62 + "\n")
    return total
