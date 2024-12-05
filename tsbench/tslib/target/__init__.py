# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck, Andreas Auer

from ...ml_utils.meta_factory import get_and_create_class_factory
from .classification import CSVClassificationTarget
from .regression import ManyToManyRegressionTarget
from .target_dataset import NoTarget

_target_generator_registry = {
    "no_target": NoTarget,
    "csv_classification": CSVClassificationTarget,
    "many_to_many_regression": ManyToManyRegressionTarget,
}


get_target_generator, create_target_generator = get_and_create_class_factory(_target_generator_registry)
