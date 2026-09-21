from __future__ import annotations

"""Trains and evaluates the SparseJointCore image classifier (see
sj_image_classifier.py) on FashionMNIST clothing classification -- a
visually harder task than MNIST, used to check whether group
specialization (see sj_image_classifier.pairwise_mi_penalty) holds up
outside of digit recognition.

Usage: python fashion_classification_test.py
Outputs (in results/fashion_mnist/): confusion_matrix.png, sample_predictions.png, metrics.txt
"""

from torchvision import datasets, transforms

from sj_image_classifier import run_classification_test

DATA_DIR = "data"
OUT_DIR = "results/fashion_mnist"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main() -> None:
    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
    run_classification_test("FashionMNIST", train_ds, test_ds, CLASS_NAMES, OUT_DIR)


if __name__ == "__main__":
    main()
