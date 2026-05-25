"""PatchNCE contrastive loss — the core of the CUT framework.

Reference: Park et al., "Contrastive Learning for Unpaired Image-to-Image
Translation", ECCV 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampleMLP(nn.Module):
    """
    2-layer MLP that projects encoder feature patches to a normalised embedding
    space where the NCE loss is computed.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


class PatchNCELoss(nn.Module):
    """
    Computes the PatchNCE loss between source patches (real_A features) and
    their corresponding translated patches (fake_B features) using the
    generator's encoder as the feature extractor.

    For each spatial location in a feature map, the positive pair is the
    (source, translated) patch at the same location; negatives are all other
    locations in the same batch element.
    """

    def __init__(self, num_patches: int = 256, temperature: float = 0.07):
        super().__init__()
        self.num_patches = num_patches
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mlps: nn.ModuleList = nn.ModuleList()

    def build_mlps(self, feat_shapes: list[tuple[int, int, int]]):
        """Call once after first forward pass when channel dims are known."""
        self.mlps = nn.ModuleList()
        for c, _, _ in feat_shapes:
            self.mlps.append(PatchSampleMLP(in_channels=c))

    def _sample_patches(self, feat: torch.Tensor):
        """Randomly sample `num_patches` spatial positions from (B, C, H, W)."""
        B, C, H, W = feat.shape
        n = min(self.num_patches, H * W)
        idx = torch.randperm(H * W, device=feat.device)[:n]
        patches = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)[:, idx, :]
        return patches, idx

    def forward(
        self,
        feats_src: list[torch.Tensor],
        feats_tgt: list[torch.Tensor],
    ) -> torch.Tensor:
        if not self.mlps or len(self.mlps) != len(feats_src):
            shapes = [(f.shape[1], f.shape[2], f.shape[3]) for f in feats_src]
            self.build_mlps(shapes)
            self.mlps = self.mlps.to(feats_src[0].device)

        total_loss = torch.tensor(0.0, device=feats_src[0].device)
        for feat_s, feat_t, mlp in zip(feats_src, feats_tgt, self.mlps):
            patches_s, idx = self._sample_patches(feat_s)
            patches_t, _   = self._sample_patches(feat_t)
            patches_t = feat_t.permute(0, 2, 3, 1).reshape(feat_t.shape[0], -1, feat_t.shape[1])[:, idx, :]

            B, N, C = patches_s.shape
            emb_s = mlp(patches_s.reshape(B * N, C)).reshape(B, N, -1)
            emb_t = mlp(patches_t.reshape(B * N, C)).reshape(B, N, -1)

            # Cosine similarity: positive on diagonal, negatives off-diagonal
            sim = torch.bmm(emb_t, emb_s.permute(0, 2, 1)) / self.temperature  # (B, N, N)
            labels = torch.arange(N, device=feat_s.device).unsqueeze(0).expand(B, -1).reshape(-1)
            total_loss += self.cross_entropy(sim.reshape(B * N, N), labels)

        return total_loss / len(feats_src)
