# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Andreas Auer

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..base.timeseries_dataset import TimeSeries
from .target_dataset import FEATURE_KEY, TARGET_KEY, TargetGenerator


@dataclass
class ManyToManyRegressionTargetConfig:
    target_features: List[str]
    input_features: List[str]
    target_shift: int


class ManyToManyRegressionTarget(TargetGenerator):
    config_class = ManyToManyRegressionTargetConfig

    def __init__(self, config: ManyToManyRegressionTargetConfig):
        self.config = config
        assert self.config.target_shift == 0, "Target Shift not implemented yet!"

    def get_target_shape(self, timeseries: TimeSeries) -> Dict[str, Tuple]:
        window_len = len(timeseries)
        # window_len -= abs(self.config.target_shift)
        return {TARGET_KEY: (window_len, len(self.config.target_features))}

    def get_input_shape(self, timeseries: TimeSeries) -> Dict[str, Tuple]:
        window_len = len(timeseries)
        # window_len -= abs(self.config.target_shift)
        return {FEATURE_KEY: (window_len, len(self.config.input_features))}

    def get_targets(self, timeseries: TimeSeries) -> np.ndarray:
        # Consider target shift
        return timeseries.dataframe[self.config.target_features].values

    def get_inputs(self, timeseries: TimeSeries) -> np.ndarray:
        # Consider target shift
        return timeseries.dataframe[self.config.input_features].values
