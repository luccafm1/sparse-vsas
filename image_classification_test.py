from __future__ import annotations

"""Trains and evaluates the SparseJointCore image classifier (see
sj_image_classifier.py) on MNIST digit classification.

Usage: python image_classification_test.py
Outputs (in results/mnist/): confusion_matrix.png, sample_predictions.png, metrics.txt
"""

from torchvision import datasets, transforms

from sj_image_classifier import run_classification_test

DATA_DIR = "data"
OUT_DIR = "results/mnist"
CLASS_NAMES = [str(d) for d in range(10)]


def main() -> None:
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    run_classification_test("MNIST", train_ds, test_ds, CLASS_NAMES, OUT_DIR)


if __name__ == "__main__":
    main()
