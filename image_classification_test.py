from __future__ import annotations

"""Proof that the SparseVQCore VSA bottleneck (base.py) is domain-agnostic:
no video, no physics, just a small CNN image encoder feeding SparseVQCore,
with a linear head reading the resulting "program" vector. Trains and
evaluates on MNIST digit classification.

Usage: python image_classification_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base import SJConfig, SparseVQCore, make_rigid_scaffold

DATA_DIR = "data"


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class ImageSJClassifier(nn.Module):
    def __init__(self, cfg: SJConfig, masks: torch.Tensor, seed: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = ImageEncoder(cfg.frame_embed)
        self.sj = SparseVQCore(cfg.frame_embed, cfg, masks, seed)
        self.head = nn.Linear(self.sj.program_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.sj(self.encoder(x), pair_writes=True)
        out["class_logits"] = self.head(out["program"])
        return out


def main() -> None:
    torch.manual_seed(0)
    cfg = SJConfig(groups=8, card=16, layers=2, units=32, topk=6, unit_width=8, hv_dim=64, frame_embed=64)
    masks = make_rigid_scaffold(cfg, seed=cfg.seed)
    model = ImageSJClassifier(cfg, masks, seed=cfg.seed, num_classes=10)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(3):
        last_loss = None
        for x, y in train_loader:
            out = model(x)
            vq, commit, kl, entropy = model.sj.vq_losses(out)
            loss = F.cross_entropy(out["class_logits"], y) + vq + 0.25 * commit + 0.05 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch + 1}/3  last batch loss={last_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)["class_logits"]
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
    print(f"test accuracy: {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
