from __future__ import annotations

"""Carregamento unificado dos datasets de imagem usados na validação.

Todos os datasets entram pelo mesmo caminho e saem no mesmo formato, para que
qualquer comparação entre eles seja feita sob protocolo idêntico:

    x            float32 [N, C*H*W] em [0, 1], achatado
    y            int64 [N]
    class_names  list[str]
    image_shape  (C, H, W)

Os splits oficiais são unidos de propósito: a validação é por k-fold
estratificado, definido em `validation.py`, e não pelo split do benchmark.
"""

from dataclasses import dataclass

import numpy as np

TORCHVISION = {
    "mnist": ("MNIST", [str(d) for d in range(10)]),
    "fashionmnist": ("FashionMNIST", ["T-shirt/top", "Trouser", "Pullover", "Dress",
                                      "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                                      "Ankle boot"]),
}

DATA_DIR = "data"


@dataclass(frozen=True)
class Dataset:
    name: str
    x: np.ndarray                 # [N, C*H*W] float32 em [0,1]
    y: np.ndarray                 # [N] int64
    class_names: list[str]
    image_shape: tuple[int, int, int]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[1])

    def imbalance_ratio(self) -> float:
        """Razão entre a classe mais e a menos frequente."""

        counts = np.bincount(self.y, minlength=self.n_classes)
        return float(counts.max() / max(counts.min(), 1))

    def describe(self) -> str:
        channels, height, width = self.image_shape
        return (f"{self.name}: {len(self.x)} amostras, {height}x{width}x{channels}, "
                f"{self.n_classes} classes, desbalanceamento {self.imbalance_ratio():.1f}x")


def _load_torchvision(name: str) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, int, int]]:
    from torchvision import datasets as tv

    cls_name, class_names = TORCHVISION[name]
    cls = getattr(tv, cls_name)
    parts = [cls(DATA_DIR, train=flag, download=True) for flag in (True, False)]

    images = np.concatenate([p.data.numpy() for p in parts], axis=0)
    labels = np.concatenate([p.targets.numpy() for p in parts], axis=0).astype(np.int64)
    if images.ndim == 3:
        images = images[..., None]
    height, width, channels = images.shape[1:]
    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    return x, labels, class_names, (channels, height, width)


def _load_medmnist(name: str) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, int, int]]:
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    if info["task"] == "multi-label, binary-class":
        raise ValueError(f"{name} é multi-label; este protocolo assume classe única")

    cls = getattr(medmnist, info["python_class"])
    parts = [cls(split=s, download=True) for s in ("train", "val", "test")]
    images = np.concatenate([p.imgs for p in parts], axis=0)
    labels = np.concatenate([p.labels for p in parts], axis=0).reshape(-1).astype(np.int64)
    if images.ndim == 3:
        images = images[..., None]
    height, width, channels = images.shape[1:]
    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return x, labels, class_names, (channels, height, width)


def load_dataset(name: str, *, sample_cap: int = 0, seed: int = 0) -> Dataset:
    """Carrega um dataset pelo nome.

    `sample_cap` subamostra de forma estratificada, preservando as proporções
    das classes. Serve para igualar o orçamento de treino entre datasets de
    tamanhos muito diferentes (MNIST tem 70k, BloodMNIST tem 17k), que é
    pré-requisito para comparar resultados entre eles.
    """

    name = name.lower()
    if name in TORCHVISION:
        x, y, class_names, shape = _load_torchvision(name)
    else:
        x, y, class_names, shape = _load_medmnist(name)

    if sample_cap and len(x) > sample_cap:
        x, y = _stratified_subsample(x, y, sample_cap, seed)

    return Dataset(name=name, x=x, y=y, class_names=class_names, image_shape=shape)


def _stratified_subsample(x: np.ndarray, y: np.ndarray, cap: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep_fraction = cap / len(x)
    keep: list[np.ndarray] = []
    for label in np.unique(y):
        index = np.flatnonzero(y == label)
        take = max(1, int(round(len(index) * keep_fraction)))
        keep.append(rng.choice(index, size=min(take, len(index)), replace=False))
    selected = np.sort(np.concatenate(keep))
    return x[selected], y[selected]


def available() -> list[str]:
    from medmnist import INFO
    two_d = [k for k, v in INFO.items()
             if not k.endswith("3d") and v["task"] != "multi-label, binary-class"]
    return sorted(TORCHVISION) + sorted(two_d)
