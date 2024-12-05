# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import pandas as pd

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset

LOGGER = logging.getLogger(__name__)


class TimeSeriesTransformsDataset(TimeSeriesDataset):
    """Class that wraps a TimeSeriesDataset and applies a list of transforms to each time series upon loading.
    Transforms are specified as a list of functions that take a TimeSeries as input and return a TimeSeries as output.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        transforms: List[Callable[[TimeSeries], TimeSeries]] = [],
    ):
        assert isinstance(transforms, list), "transforms must be a list of callables."
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index) -> TimeSeries:
        ts = self.dataset[index]
        for transform in self.transforms:
            ts = transform(ts)
        return ts

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def root(self):
        return self.dataset.root

    def get_meta_data_summary(
        self, meta_data_types: Tuple[Type] = (int, str, float), indices: List[int] = [], **kwargs
    ) -> pd.DataFrame:
        return self.dataset.get_meta_data_summary(meta_data_types, indices, **kwargs)


@dataclass
class FeatureSelectorConfig:
    select_features: Optional[List[str]] = field(default_factory=list)
    drop_features: Optional[List[str]] = field(default_factory=list)


class FeatureSelector:
    """Selects or drops features from a time series.
    If both select_features and drop_features are empty, this is a no-op.

    Args:
        select_features: List of features to select. If empty, all features are selected.
        drop_features: List of features to drop. If empty, no features are dropped.
    """

    config_class = FeatureSelectorConfig

    def __init__(self, config: FeatureSelectorConfig):
        self.config = config
        # XOR
        assert isinstance(self.config.select_features, list) and isinstance(
            self.config.drop_features, list
        ), "select_features and drop_features must be lists."
        # Disable warning for chained assignment
        pd.options.mode.chained_assignment = None  # default='warn'

    def __call__(self, timeseries: TimeSeries) -> TimeSeries:
        if self.config.select_features or self.config.drop_features:
            if self.config.drop_features:
                drop_features = self.config.drop_features
            else:
                assert set(self.config.select_features).issubset(
                    set(timeseries.features)
                ), "select_features contains features that are not present in the time series."
                drop_features = list(set(timeseries.features) - set(self.config.select_features))

            timeseries.dataframe.drop(columns=drop_features, inplace=True)
        return timeseries


@dataclass
class NormalizerConfig:
    normalizer_values: dict[str, Dict[str, float]] = field(default_factory=dict)
    normalize_features: set[str] = field(default_factory=set)
    drop_zero_variance_features: bool = True
    eps: float = 1e-8
    normalizer_file: Optional[str] = None


class Normalizer:
    """
    Normalizes (selected) features of a time series.

    Useful reference: https://www.geeksforgeeks.org/normalize-a-column-in-pandas/

    Args:
        normalizer_values: Dictionary of normalizer values for each feature. The dictionary must have the following structure:
            { "mean" : { "feature1" : 0.0, "feature2" : 0.0, ... }, "std" : { "feature1" : 1.0, "feature2" : 1.0, ... } }
            If a feature is not present in the dictionary, it is not normalized.
            Features / colum names present in the dictionary but not in the time series are ignored.
        normalize_features: List of features to normalize. If empty, all features are normalized.
        drop_zero_variance_features: If True, columns with zero variance are dropped. If False, they are kept and not normalized.

    """

    config_class = NormalizerConfig

    def __init__(self, config: NormalizerConfig):
        self.config = config
        if self.config.normalizer_file is not None:
            with open(self.config.normalizer_file) as f:
                self.config.normalizer_values = json.load(f)

        # make the Normalizer a no-op if no normalizer values are provided
        if self.config.normalizer_values:
            assert isinstance(self.config.normalizer_values, dict), "normalizer_values must be a dictionary."
            assert set(self.config.normalizer_values["mean"].keys()) == set(
                self.config.normalizer_values["std"].keys()
            ), "normalizer_values must contain the same features for mean and std."
            assert set(self.config.normalize_features).issubset(
                set(self.config.normalizer_values["mean"].keys())
            ), "normalize_features contains features that are not present in normalizer_values."

            # get drop features with zero variance
            self.drop_features = []
            if self.config.drop_zero_variance_features:
                self.drop_features = self._get_zero_variance_features(self.config.normalizer_values)
                if self.drop_features:
                    LOGGER.info(f"Dropping {len(self.drop_features)} zero variance features: {self.drop_features}")
                    print(f"Dropping {len(self.drop_features)} zero variance features: {self.drop_features}")
                    pd.options.mode.chained_assignment = None  # default='warn'

            # get selected features mean and std
            self.modified_normalized_features = self.config.normalize_features
            if len(self.modified_normalized_features) == 0:
                self.modified_normalized_features = list(
                    set(self.config.normalizer_values["mean"].keys()) - set(self.drop_features)
                )

            self._normalize_features_mean = pd.Series(self.config.normalizer_values["mean"])[
                self.modified_normalized_features
            ]
            self._normalize_features_std = pd.Series(self.config.normalizer_values["std"])[
                self.modified_normalized_features
            ]

            # cache the features to normalize features
            self._features_to_normalize = None

    def _get_zero_variance_features(self, normalizer_values: Dict[str, Dict[str, float]]) -> List[str]:
        std_ss = pd.Series(normalizer_values["std"])
        return list(std_ss.index[std_ss == 0.0])

    def _normalize_selected(self, dataframe: pd.DataFrame, mean_ss: pd.Series, std_ss: pd.Series) -> pd.DataFrame:
        if self._features_to_normalize is None:
            self._features_to_normalize = mean_ss.index.intersection(dataframe.columns)
        dataframe[self._features_to_normalize] = (
            dataframe[self._features_to_normalize] - mean_ss[self._features_to_normalize]
        ) / (std_ss[self._features_to_normalize] + self.config.eps)
        return dataframe

    def __call__(self, timeseries: TimeSeries) -> TimeSeries:
        # make the Normalizer a no-op if no normalizer values are provided
        if self.config.normalizer_values:
            if self.drop_features:
                timeseries.dataframe.drop(columns=self.drop_features, inplace=True)
            if self.modified_normalized_features:
                df = timeseries.dataframe
                # normalize selected features
                df = self._normalize_selected(df, self._normalize_features_mean, self._normalize_features_std)
                timeseries.set_dataframe(df)

        return timeseries


@dataclass
class MissingValueHandlerConfig:
    features: List[str] = field(default_factory=list)
    fillmethod: str = "nofill"  # nofill, forwardfill, backwardfill, interpolate
    kwargs: Dict[str, Any] = field(default_factory=dict)
    add_missing_value_indicator: bool = True
    missing_value_indicator_suffix: str = "--NaN"

    def __post_init__(self):
        fillmethods = ["nofill", "forwardfill", "backwardfill", "meanfill", "constfill", "interpolate"]
        assert self.fillmethod in fillmethods, f"fillmethod must be one of {fillmethods}, but is {self.fillmethod}"


## Fill implementations:
_fillmethod_fillnamethod_map = {"forwardfill": "ffill", "backwardfill": "bfill"}


# implementations are splitted by the pandas fill methods used
def _nofill(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df


def _fillna(df: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    if method == "meanfill":
        return df.fillna(df.mean())
    elif method == "constfill":
        const_value = kwargs.pop("const_value")
        return df.fillna(const_value)
    else:
        method = _fillmethod_fillnamethod_map[method]
        return df.fillna(method=method)


def _interpolate(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df.interpolate(**kwargs)


class MissingValueHandler:
    """
    This transform handles missing values in the time series. It replaces missing values indicated by NaN.
    The transform can be configured to fill missing values with a specific method, e.g. forward fill.
    The transform can also add a missing value indicator feature for each feature that is handled, i.e. was NaN.

    If one wants to specify different fill methods for different features, one can use multiple MissingValueHandler.

    Args:
        features: The features to handle. If empty, all features are handled.
        fillmethod: The method to fill missing values. One of "nofill", "forwardfill", "backwardfill", "meanfill", "constfill", "interpolate".
        kwargs: Keyword arguments passed to the fill method.
        add_missing_value_indicator: If True, a missing value indicator feature is added for each feature that is handled.
        missing_value_indicator_suffix: The suffix of the missing value indicator feature.
    """

    config_class = MissingValueHandlerConfig

    def __init__(self, config: MissingValueHandlerConfig):
        self.config = config

    def __call__(self, timeseries: TimeSeries) -> TimeSeries:
        features_to_handle = self.config.features
        if len(features_to_handle) == 0:
            features_to_handle = timeseries.features

        df = timeseries.dataframe
        df_selected_fill = df[features_to_handle]

        # add missing value indicator
        missing_value_indicator_df = pd.DataFrame({})
        if self.config.add_missing_value_indicator:
            missing_value_indicator_df = (
                df_selected_fill.isna().astype(float).add_suffix(self.config.missing_value_indicator_suffix)
            )

        # fill missing values
        if self.config.fillmethod == "nofill":
            df_selected_fill = _nofill(df_selected_fill, **self.config.kwargs)
        elif self.config.fillmethod in ["forwardfill", "backwardfill", "meanfill", "constfill"]:
            df_selected_fill = _fillna(df_selected_fill, self.config.fillmethod, **self.config.kwargs)
        elif self.config.fillmethod == "interpolate":
            df_selected_fill = _interpolate(df_selected_fill, **self.config.kwargs)

        df[features_to_handle] = df_selected_fill

        df = pd.concat([df, missing_value_indicator_df], axis=1)

        timeseries.set_dataframe(df)
        return timeseries
