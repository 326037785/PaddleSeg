"""Additional usage examples to help practitioners transition to PaddleSeg.

These tests intentionally mirror common coding patterns from other deep
learning frameworks (such as PyTorch) so that beginners can quickly map their
existing knowledge to the PaddleSeg API surface.  They run on small random
inputs to keep execution light-weight while still exercising the most common
entry points of the library.
"""

import numpy as np
import paddle

from paddle.io import DataLoader

from paddleseg.models import UNet
from paddleseg.transforms import Compose, Normalize, Resize


class TorchStyleToyDataset(paddle.io.Dataset):
    """A toy dataset that mimics the torch.utils.data.Dataset interface.

    Each sample is generated on the fly so no external assets are required.
    The dataset applies a PaddleSeg ``Compose`` transform internally, which is
    conceptually similar to torchvision's transform pipelines.  Returning
    tensors keeps the integration with ``paddle.io.DataLoader`` familiar for
    users migrating from PyTorch.
    """

    def __init__(self,
                 num_samples=4,
                 num_classes=2,
                 image_size=(48, 48),
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.height, self.width = image_size
        self.transform = Compose(
            [
                Resize(target_size=(self.width, self.height)),
                Normalize(mean=mean, std=std),
            ],
            to_rgb=False,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx  # Unused for synthetic data.
        image = (np.random.rand(self.height, self.width, 3).astype("float32") *
                 255.0)
        label = np.random.randint(
            low=0,
            high=self.num_classes,
            size=(self.height, self.width),
            dtype="int64",
        )
        sample = self.transform({"img": image, "label": label})
        image_tensor = paddle.to_tensor(sample["img"], dtype="float32")
        label_tensor = paddle.to_tensor(sample["label"], dtype="int64")
        return image_tensor, label_tensor


def test_compose_pipeline_feels_familiar_for_torch_users():
    """Showcase how PaddleSeg transforms mirror torchvision style pipelines."""
    dataset = TorchStyleToyDataset(num_samples=1, image_size=(32, 32))
    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == paddle.float32
    assert label.shape == (32, 32)
    assert label.dtype == paddle.int64


def test_one_training_step_matches_common_deep_learning_pattern():
    """Demonstrate a training step that resembles a PyTorch lightning loop."""
    paddle.disable_static()
    paddle.set_device("cpu")

    dataset = TorchStyleToyDataset(num_samples=2, image_size=(32, 32))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = UNet(num_classes=2, in_channels=3)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=0.001)
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    for images, labels in dataloader:
        optimizer.clear_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        # The returned scalar tensor makes it easy to log just like in PyTorch.
        assert loss.shape == []
        break
