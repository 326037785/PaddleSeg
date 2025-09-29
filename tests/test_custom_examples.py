"""High level integration examples requested by community users."""

import pytest

paddle = pytest.importorskip("paddle", reason="paddlepaddle is not installed")
import paddle.nn.functional as F

from paddleseg.models import UNet
from paddleseg.models.losses import DiceLoss
from paddleseg.models.layers.layer_libs import ConvBNReLU


class HybridSegmentationLoss(paddle.nn.Layer):
    """Blend cross-entropy and dice losses with class-wise reweighting."""

    def __init__(self,
                 num_classes,
                 ce_weight=0.6,
                 dice_weight=0.4,
                 class_weights=None,
                 ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            weight_tensor = paddle.to_tensor(class_weights, dtype="float32")
            assert weight_tensor.shape[0] == num_classes, (
                "The number of class weights must match num_classes.")
            self.register_buffer("class_weights", weight_tensor)
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        ce_term = F.cross_entropy(logits,
                                  labels,
                                  ignore_index=self.ignore_index,
                                  weight=self.class_weights,
                                  reduction='mean')
        dice_term = self.dice_loss(logits, labels)
        total = self.ce_weight * ce_term + self.dice_weight * dice_term
        return total


def test_custom_hybrid_loss_example():
    """Run a forward pass with a UNet and the hybrid loss."""
    paddle.disable_static()
    paddle.seed(123)

    model = UNet(num_classes=3, in_channels=3)
    loss_fn = HybridSegmentationLoss(num_classes=3,
                                     ce_weight=0.7,
                                     dice_weight=0.3,
                                     class_weights=[1.0, 2.0, 3.0])

    inputs = paddle.randn([2, 3, 64, 64], dtype="float32")
    labels = paddle.randint(low=0,
                            high=3,
                            shape=[2, 64, 64],
                            dtype="int64")

    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss_value = float(loss.numpy())
    print(f"Hybrid segmentation loss: {loss_value}")
    assert loss.ndim == 0
    assert loss_value > 0


class TinyCustomSegmentor(paddle.nn.Layer):
    """A lightweight segmentor made from PaddleSeg building blocks."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = ConvBNReLU(3, 16, 3)
        self.encoder = paddle.nn.Sequential(
            ConvBNReLU(16, 32, 3),
            ConvBNReLU(32, 32, 3),
        )
        self.classifier = paddle.nn.Conv2D(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def test_custom_component_model_forward_backward():
    """Showcase a forward/backward pass with a custom model class."""
    paddle.disable_static()
    paddle.seed(2023)

    model = TinyCustomSegmentor(num_classes=4)
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=0.01,
                                          momentum=0.9)

    inputs = paddle.randn([2, 3, 32, 32], dtype="float32")
    labels = paddle.randint(low=0,
                            high=4,
                            shape=[2, 32, 32],
                            dtype="int64")

    optimizer.clear_grad()
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    loss_value = float(loss.numpy())
    print(f"Custom component network loss: {loss_value}")
    assert loss.ndim == 0
    assert loss_value > 0
