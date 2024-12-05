# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import sys
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm


def calculate_dataset_mean_std(
    dataset: data.Dataset, batch_size: int = 256, num_workers: int = 6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the mean and std of a given dataset.

    Args:
        dataset (data.Dataset): The dataset.
        batch_size (int, optional): The batch size. Defaults to 256.

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): the mean and std of the dataset along the second dimension (first is batch dimension)
    """
    dl = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    mean = 0.0
    std = 0.0
    num_samples = 0
    for x, y in tqdm(dl, file=sys.stdout):
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)  # keep channel dim
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std


def show_images(
    img_dataset: Union[
        data.Dataset, torch.Tensor, np.ndarray, Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]
    ],
    num_images: int,
    n_cols: int = 4,
    shuffle: bool = False,
    cmap: str = None,
    select_idxes: List[int] = [],
    label_names: List[str] = [],
    fname: str = "",
):
    if isinstance(img_dataset, (torch.Tensor, np.ndarray)):
        if isinstance(img_dataset, np.ndarray):
            img_dataset = torch.from_numpy(img_dataset)
        img_dataset = data.TensorDataset(img_dataset, torch.ones(img_dataset.shape[0]) * (-1))
    if isinstance(img_dataset, tuple):
        assert len(img_dataset) == 2
        if isinstance(img_dataset[0], np.ndarray):
            img_dataset = [torch.from_numpy(d) for d in img_dataset if isinstance(d, (np.ndarray))]
        img_dataset = data.TensorDataset(*img_dataset)

    if select_idxes:
        idxes = select_idxes
        num_images = len(idxes)
    else:
        idxes = np.arange(len(img_dataset))

    if shuffle:
        np.random.default_rng().shuffle(idxes)

    n_rows = num_images // n_cols
    plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        if shuffle or select_idxes:
            idx = idxes[i]
        else:
            idx = i
        img = img_dataset[idx][0]
        label = img_dataset[idx][1]
        if isinstance(label, (torch.Tensor, np.ndarray)):
            label = label.item()
        if label_names and label >= 0:
            label = label_names[label]
        if isinstance(img, (torch.Tensor, np.ndarray)):
            plt.imshow(img.permute(1, 2, 0), cmap=cmap)
        elif isinstance(img, Image.Image):
            plt.imshow(img, cmap=cmap)
        else:
            raise ValueError(f"Unknown image type: {type(img)}")
        plt.title(f"Label: {label}")
        plt.axis("off")
        if i >= len(img_dataset) - 1:
            break
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()


def show_image_grid(
    images: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
):
    import torchvision.transforms.functional as F
    from torchvision.utils import make_grid

    imgs = make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value,
    )

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
