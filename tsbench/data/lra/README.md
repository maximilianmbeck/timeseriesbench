# Long Range Arena Datasets

Long Range Arena consists of 6 tasks:

- ListOps:

  Modeling hierarchically structured data in a long-context scenario.
  Fixed max sequence length of up to 2K (2048), padded when necessary.
  Ten-way classification task.

  Dataset in .tsv format.

- Text:

  byte-level text classification on the IMDb reviews dataset.
  Fixed max length of 4K (truncated or padded when necessary).
  Two-way (binary) classification task.

  Dataset has to be downloaded separately. Load from Huggingface: https://huggingface.co/datasets/imdb

- Retrieval:

  byte-level text retrieval on the ACL Anthology Network dataset, which identifies if two papers have a citation link.
  Use a sequence length of 4K for each document -> total length of 8K.
  Two-way (binary) classification task.

  Dataset in .tsv format provided in the `lra_release` folder.

- Image (Sequential grayscale CIFAR-10):

  Map the input images to a single gray-scale channel where each pixel is represented in 8bit (vocabulary size of 256).
  LRA use CIFAR10.
  Ten-way classification task.

  Dataset has to be downloaded from torchvision.

- Pathfinder:

  The task requires a model to make a binary decision whether two points represented as circles are connected by a path consisting of dashes.
  Images are of size 32x32, which yields a sequence length of 1024.
  Two-way (binary) classification task.

  Dataset in .png format in the `lra_release` folder.

- Path-X:

  Pathfinder (Pathfinder-X) with larger image size (128x128) and longer sequence length (16K).
  Two-way (binary) classification task.

  Dataset in .png format in the `lra_release` folder.


## Data format

TODO process dataset to be in the following format (do it automatically):

```
$DATA_PATH/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
```

### Dev notes

Implementing Cifar lra:

config arguments in safari repo:
```yaml
_name_: cifar
permute: null
grayscale: True
tokenize: False
augment: False
cutout: False
random_erasing: False
val_split: 0.1
seed: 42 # For validation split
# __l_max: 1024
```

In their code they have lots of optional transforms and permutations. We will ignore them for now and just use the grayscale and resize to 32x32.

There is also the option to tokenize the images: Tokenization converts the images to a sequence of integers in 8bit, i.e. 256 tokens.
In their experiments the do not tokenize the images, but instead use the normalized grayscale values as input and use a linear layer instead of an embedding layer.
