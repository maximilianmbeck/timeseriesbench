# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..base.timeseries_dataset import TimeSeries
from ..loading.csv_loader import CSVTimeSeries, CSVTimeSeriesMeta
from ..postprocessing.window_dataset import TimeSeriesWindow
from .target_dataset import TARGET_KEY, TargetGenerator


class BaseClassificationTarget(TargetGenerator):
    """Generic Target generator for classification.
    Computes the target by applying a function to the time series.
    The function returns the class label for the given time series.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def get_target_shape(self, timeseries: TimeSeries) -> dict[str, tuple]:
        return {TARGET_KEY: (self.num_classes,)}

    @abstractmethod
    def get_targets(self, timeseries: Union[TimeSeries, TimeSeriesWindow]) -> np.ndarray:
        pass


@dataclass
class ClassificationTargetConfig:
    # this list contains the class labels in the order of the class indices
    # the index in the list corresponds to the class index
    class_labels: list[str]


class ClassificationTarget(BaseClassificationTarget):
    config_class = ClassificationTargetConfig

    def __init__(self, config: ClassificationTargetConfig):
        self.config = config
        assert isinstance(self.config.class_labels, list), "class_labels must be a list"
        assert len(self.config.class_labels) > 0, "class_labels must not be empty"

        self._classidx_to_classlabel = {
            class_label: class_idx for class_idx, class_label in enumerate(self.config.class_labels)
        }

        super().__init__(num_classes=len(self.config.class_labels))

    @abstractmethod
    def _get_class_label(self, timeseries: TimeSeries) -> str:
        """Return the class label for the given time series.
        This must be specific to the dataset, since the class labels can be encoded in different ways
        for different datasets."""
        pass

    def get_targets(self, timeseries: Union[TimeSeries, TimeSeriesWindow]) -> np.ndarray:
        return np.array(self._classidx_to_classlabel[self._get_class_label(timeseries)])


@dataclass
class CSVClassificationTargetConfig(ClassificationTargetConfig):
    class_column: str


class CSVClassificationTarget(ClassificationTarget):
    """Target generator for classification that uses the subject column from the csv file."""

    config_class = CSVClassificationTargetConfig

    def __init__(self, config: CSVClassificationTargetConfig):
        super().__init__(config)
        self.config = config

    def _get_class_label(self, timeseries: CSVTimeSeries) -> str:
        return timeseries.meta_data.csv_meta[self.config.class_column]


#! Just for reference, another example of a classification target:
# @dataclass
# class VehicleClassificationTargetConfig:
#     vehicle_ids: List[int]


# class VehicleClassificationTarget(ClassificationTarget):

#     config_class = VehicleClassificationTargetConfig

#     def __init__(self, config: VehicleClassificationTargetConfig):
#         self.config = config
#         assert isinstance(self.config.vehicle_ids, list), "vehicle_ids must be a list"
#         assert len(self.config.vehicle_ids) > 0, "vehicle_ids must not be empty"

#         def _label_function(timeseries: TimeSeries) -> int:
#             return self._get_driver_class(timeseries)

#         super().__init__(num_classes=len(self.config.vehicle_ids), label_function=_label_function)

#         # build mapping from driver id to class label
#         self._driver_id_to_class_label = {driver_id: class_label for class_label, driver_id in enumerate(self.config.vehicle_ids)}

#     def _get_driver_class(self, timeseries: TimeSeries) -> int:
#         if isinstance(timeseries, TimeSeriesWindow):
#             timeseries = timeseries.timeseries
#         assert isinstance(timeseries, DrivingTimeSeries), "timeseries must be a DrivingTimeSeries"
#         return self._driver_id_to_class_label[timeseries.vehicle_id]
