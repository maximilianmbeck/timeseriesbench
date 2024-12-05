# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torchmetrics
from dacite import Config as DaciteConfig
from dacite import from_dict
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.functional.text.perplexity import _check_shape_and_type_consistency
from torchmetrics.metric import Metric

from .config import MetricConfig
from .ml_utils.data.datasetgeneratorinterface import DatasetGeneratorInterface

# A unit test for a metric must contain the following:
# - test creation
# - test forward, (values are computed correctly)
# - test reset
# - test move to device


class SequenceAccuracy(Metric):
    """
    >>> LastOutputAccuracy

    """

    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, **kwargs):
        super().__init__()
        self._acc = torchmetrics.Accuracy(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape((-1, preds.shape[-1]))
        target = target.flatten()
        return self._acc.update(preds, target)

    def compute(self):
        return self._acc.compute()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._acc.reset()


def create_accuracy_metric(num_classes: int, **kwargs) -> Metric:
    if num_classes == 2:
        task = "binary"
    elif num_classes > 2:
        task = "multiclass"
    else:
        raise ValueError(f"Invalid output dimension: {num_classes}.")
    return torchmetrics.Accuracy(num_classes=num_classes, task=task, **kwargs)


@dataclass
class BucketPerplexityCollectionConfig:
    token_count_file: str
    bucket_ranges: Sequence[Union[float, int, str]]
    split: str = "train"


class BucketPerplexityCollection:
    """Generates a collection of BucketPerplexity metrics for a given set of buckets."""

    # Note: Another possibility to create this collection would be to replace __new__ with __call__ and
    #       register an BucketPerplexityCollection() object in the _metric_registry. (staticmethods can then be regular methods)
    #       From the view of the `create_metrics()` function this is equivalent to the current implementation.

    config_class = BucketPerplexityCollectionConfig

    @staticmethod
    def get_bucket_name(r: tuple[int, int]) -> str:
        return f"ppl_bucket_cts_{r[0]}-{r[1]}"

    @staticmethod
    def generate_ranges(
        buckets: Sequence[Union[float, int, str]]
    ) -> dict[str, tuple[Union[int, str], Union[int, str]]]:
        def _convert_to_int(x: Union[float, int, str]) -> Union[int, str]:
            if not type(x) is str:
                return int(x)
            else:
                return x

        bucket_ranges = []
        for i in range(len(buckets)):
            lower = _convert_to_int(buckets[i])

            if (i + 1) < len(buckets):
                upper = _convert_to_int(buckets[i + 1])
                if type(upper) is not str:
                    upper -= 1
            else:
                break
            bucket_ranges.append((lower, upper))

        named_buckets = {}
        for bucket in bucket_ranges:
            named_buckets[BucketPerplexityCollection.get_bucket_name(bucket)] = bucket

        return named_buckets

    def __new__(cls, config: BucketPerplexityCollectionConfig) -> torchmetrics.MetricCollection:
        bucket_ranges = cls.generate_ranges(config.bucket_ranges)

        token_count_file = Path(config.token_count_file)
        if not token_count_file.exists():
            raise ValueError(f"Token count file {token_count_file} does not exist.")

        # load token counts only once and pass dict to each BucketPerplexity metric
        with open(token_count_file, "r") as f:
            token_counts = json.load(f)
            assert (
                config.split in token_counts
            ), f"Did not find token counts for {config.split} in {config.token_count_file}"

        perplexities = {}
        for bucket in bucket_ranges:
            perplexities[bucket] = BucketPerplexity(
                token_counts=token_counts, split=config.split, bucket_range=bucket_ranges[bucket]
            )

        return torchmetrics.MetricCollection(perplexities)


class BucketPerplexity(Metric):
    r"""Bucket perplexity measures how well a language model predicts the tokens that appear
    in a given range frequency range.

    It's calculated as the average number of bits per word a model needs to represent the sample
    with the respective bucket.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Logits or a unnormalized score assigned to each token in a sequence with shape
        [batch_size, seq_len, vocab_size], which is the output of a language model. Scores will be normalized internally
        using softmax.
    - ``target`` (:class:`~torch.Tensor`): Ground truth values with a shape [batch_size, seq_len]

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``perp`` (:class:`~torch.Tensor`): A tensor with the bucket perplexity score

    Args:
        bucket_range: A tuple of lower and upper token counts used for the bucket. If 'min' is
                      given as lower bound the minimum value found in the token counts. For
                      the upper bound 'max' may used to get the maximum value of token counts.
        token_count_file: Path to the json-file containing the tokens counts within the train, test, and
                  validation sets respectively
        token_counts: A dictionary containing the token counts for each dataset split.
                      Either pass this or the token_count_file. Mainly used to avoid reloading the file, if multiple
                      BucketPerplexity metrics are used.
        dset_split: Specifies the dataset split in which the tokens have been counted.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    """

    is_differentiable = False
    higher_is_better = False
    total_log_probs: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        bucket_range: tuple[Union[float, int, str], Union[float, int, str]],
        token_counts: Optional[dict[str, Sequence[int]]] = None,
        token_count_file: Optional[Path] = None,
        dset_split: str = "train",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        if token_counts is None and token_count_file is None:
            raise ValueError("Either token_counts or token_count_file must be passed.")

        if token_counts is None:
            with open(token_count_file, "r") as f:
                token_counts = json.load(f)
                assert dset_split in token_counts, f"Did not find token counts for {dset_split} in {token_count_file}"

        token_counts = token_counts[dset_split]

        # register bucket as buffer, such that it is moved to the correct device with .to()
        bucket = self._get_bucket(token_counts, bucket_range)
        self.register_buffer("bucket", bucket)

        self.add_state("total_log_probs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _get_bucket(
        self, token_counts: Sequence[int], bucket_range: tuple[Union[int, str], Union[int, str]]
    ) -> torch.Tensor:
        bucket = []

        lower_bound, upper_bound = bucket_range

        if lower_bound == "min":
            lower_bound = min(token_counts)
        if upper_bound == "max":
            upper_bound = max(token_counts)

        # should be integers, but user might call with floats
        lower_bound = int(lower_bound)
        upper_bound = int(upper_bound)

        assert lower_bound <= upper_bound, f"Minimum value of bucket must be given first. Got {bucket_range}"
        assert lower_bound >= 0, "Bucket range can not be negative"

        for token, count in enumerate(token_counts):
            if count >= lower_bound and count <= upper_bound:
                bucket.append(token)

        assert (
            bucket
        ), f"Bucket is empty. Please choose proper bucket range (token counts range form {min(token_counts)} to {max(token_counts)})"
        return torch.tensor(bucket)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""

        _check_shape_and_type_consistency(preds, target)

        probs = F.softmax(preds.reshape(-1, preds.shape[-1]), dim=1)
        target = target.reshape(-1)
        mask = torch.isin(target, self.bucket)
        probs = probs[:, target].diagonal()[mask]
        total_log_probs = -probs.log().sum()
        count = mask.sum()

        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self) -> torch.Tensor:
        """Compute the Perplexity."""
        return torch.exp(self.total_log_probs / self.count)


_metric_registry = {
    "sequence_accuracy": SequenceAccuracy,
    "bucket_perplexity": BucketPerplexity,
    "bucket_perplexity_collection": BucketPerplexityCollection,
}


def get_metric(name: str) -> Callable[[Any], Union[Metric, MetricCollection]]:
    if name in _metric_registry:
        return _metric_registry[name]
    else:
        raise ValueError(f"Metric {name} not found. Available metrics are {_metric_registry.keys()}")


def create_metrics(name: str, kwargs: dict[str, Any]) -> Union[Metric, MetricCollection]:
    metric = get_metric(name)

    metric_cfg_cls = getattr(metric, "config_class", None)
    if metric_cfg_cls is not None:
        metric_cfg = from_dict(data_class=metric_cfg_cls, data=kwargs, config=DaciteConfig(strict=True))
        return metric(metric_cfg)
    else:
        return metric(**kwargs)


def get_metrics_getter(
    metric_cfgs: Sequence[MetricConfig],
    datasetgenerator: DatasetGeneratorInterface,
) -> Callable[[], tuple[MetricCollection, MetricCollection]]:
    """Returns a function that extracts the metrics from a datasetgenerator.

    Args:
        datasetgenerator (DatasetGeneratorInterface): A datasetgenerator.

    Returns:
        Callable[[], tuple[MetricCollection, MetricCollection]]: A function that returns the metrics of the datasetgenerator.
    """
    train_metrics = []
    val_metrics = []
    for metric_cfg in metric_cfgs:
        if "train" in metric_cfg.stage:
            train_metrics.append(create_metrics(name=metric_cfg.name, kwargs=metric_cfg.kwargs))
        if "validation" in metric_cfg.stage:
            val_metrics.append(create_metrics(name=metric_cfg.name, kwargs=metric_cfg.kwargs))

    train_metrics = MetricCollection(train_metrics)
    val_metrics = MetricCollection(val_metrics)

    def get_metrics() -> MetricCollection:
        return MetricCollection([datasetgenerator.train_metrics, train_metrics]), MetricCollection(
            [datasetgenerator.validation_metrics, val_metrics]
        )

    return get_metrics


def get_metrics(
    metric_cfgs: Sequence[MetricConfig],
    datasetgenerator: DatasetGeneratorInterface,
) -> tuple[MetricCollection, MetricCollection]:
    """Returns train and validation metrics.

    Args:
        metric_cfgs (Sequence[MetricConfig]): A list of metric configs.
        datasetgenerator (DatasetGeneratorInterface): A datasetgenerator.

    Returns:
        tuple[MetricCollection, MetricCollection]: Train and validation metrics.
    """
    train_metrics = []
    val_metrics = []
    for metric_cfg in metric_cfgs:
        if "train" in metric_cfg.stage:
            train_metrics.append(create_metrics(name=metric_cfg.name, kwargs=metric_cfg.kwargs))
        if "validation" in metric_cfg.stage:
            val_metrics.append(create_metrics(name=metric_cfg.name, kwargs=metric_cfg.kwargs))

    train_metrics = MetricCollection(train_metrics)
    val_metrics = MetricCollection(val_metrics)

    return MetricCollection([datasetgenerator.train_metrics, train_metrics]), MetricCollection(
        [datasetgenerator.validation_metrics, val_metrics]
    )
