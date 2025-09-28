"""Test script for running HRNet backbone with OCRNet head on fake data.

This script builds an OCRNet model with an HRNet backbone, creates two pairs of
fake image/label tensors, and runs a single epoch of training. The goal is to
provide a lightweight utility for stress-testing GPU memory consumption without
requiring any pretrained weights or external datasets.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import paddle

from paddleseg.models import OCRNet
from paddleseg.models.backbones import HRNet_W18


class FakeSegDataset:
    """A minimal iterable dataset that yields random segmentation samples."""

    def __init__(self, num_samples, num_classes, image_shape):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_shape = image_shape

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            image = paddle.randn(self.image_shape, dtype="float32")
            label_shape = (self.image_shape[0], ) + self.image_shape[2:]
            label = paddle.randint(
                low=0,
                high=self.num_classes,
                shape=label_shape,
                dtype="int64",
            )
            yield image, label


def main():
    paddle.disable_static()
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)

    num_classes = 2
    image_shape = (1, 3, 256, 256)
    dataset = FakeSegDataset(num_samples=2,
                             num_classes=num_classes,
                             image_shape=image_shape)

    backbone = HRNet_W18(pretrained=None)
    model = OCRNet(num_classes=num_classes,
                   backbone=backbone,
                   backbone_indices=(0, ),
                   pretrained=None)
    model.train()

    optimizer = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters())
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=255, axis=1)

    for epoch in range(1):
        for step, (images, labels) in enumerate(dataset):
            optimizer.clear_grad()
            logits = model(images)
            main_loss = criterion(logits[0], labels)
            loss = main_loss
            if len(logits) > 1:
                aux_loss = criterion(logits[1], labels)
                loss = loss + 0.4 * aux_loss

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, step {step + 1}: loss={float(loss.numpy())}")


if __name__ == "__main__":
    main()
