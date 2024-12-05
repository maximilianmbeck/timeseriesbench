# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Andreas Auer

import operator
from functools import reduce
from typing import Callable, List


def map_identity():
    return lambda x: x


def merge_dicts(key_map: dict = None, value_map_func: Callable = None):
    if key_map is None and value_map_func is None:
        return lambda *x: reduce(operator.ior, x, {})
    elif key_map is None:
        return lambda *x: {key: value_map_func(val) for key, val in reduce(operator.ior, x, {}).items()}
    elif value_map_func is None:
        return lambda *x: {key_map[key]: val for key, val in reduce(operator.ior, x, {}).items() if key in key_map}
    else:
        return lambda *x: {
            key_map[key]: value_map_func(val) for key, val in reduce(operator.ior, x, {}).items() if key in key_map
        }


def map_dict_but(exclude: List[str]):
    return lambda data_dict: {key: val for key, val in data_dict.items() if key not in exclude}


def batch_to_device_all():
    return lambda batch_data, device: {key: val.to(device) for key, val in batch_data.items()}


def batch_to_device_all_but(exclude: List[str]):
    return lambda batch_data, device: {key: val.to(device) for key, val in batch_data.items() if key not in exclude}
