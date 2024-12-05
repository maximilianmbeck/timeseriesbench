# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from ..ml_utils.time_utils import Stopwatch
from .base.timeseries_dataset import TimeSeriesDataset

# add a function to create a datasplit
# split should be evenly across vehicles (probably by snippet count (or by num timesteps))
# use: from sklearn.model_selection import StratifiedShuffleSplit


def benchmark_dataloading(tsdataset: TimeSeriesDataset, num_epochs: int = 3) -> None:
    epoch_times = []
    with Stopwatch() as sw:
        for i in tqdm(range(num_epochs), desc="Epoch", file=sys.stdout):
            for j in tqdm(range(len(tsdataset)), desc="Sample", file=sys.stdout):
                tsdataset[j]
            epoch_times.append(sw.lap())

    print(f"Dataloading benchmark for {tsdataset.__class__.__name__}")
    print("Time in seconds")
    print(f"Time per epoch (num_epochs={num_epochs}): {epoch_times}")
    print(f"Average time per epoch: {sum(epoch_times) / len(epoch_times)}")
    print(f"Total time: {sw.elapsed_seconds}")
    print(f"Time first epoch: {epoch_times[0]} / Time last epoch: {epoch_times[-1]}")


def subset_assignment(
    input_cost_per_item: np.ndarray, n_splits: int, cost_func: Callable[[np.ndarray], np.ndarray] = np.abs
) -> Tuple[Dict[int, List[int]], Dict[int, Union[int, float]], Union[int, float]]:
    """Assign items to subsets such that the number of items per subset is as close as possible to the same number.
    Use the Hungarian algorithm to solve the linear sum assignment problem.
    The assignment is computed iteratively.
    In each iteration, the cost of assigning each item to each subset is computed and the optimal assignment is used.
    Therefore, with each iteration the number of items per subset increases if there are still items left to assign.

    Args:
        input_cost_per_item (np.ndarray): 1D array of costs (e.g. number of samples) per item
        n_splits (int): number of subsets
        cost_func (Callable, optional): A cost function. Defaults to np.abs.

    Returns:
        Tuple[Dict[int, List[int]], Dict[int, Union[int, float]], Union[int, float]]: assignment, current_assignment_cost, target_n_items_per_split

    Example:

    ```python
    example = np.array([10, 15, 5, 20, 30, 2, 3])
    subset_ass, num_samples, tgt_n_items = subset_assignment(example, n_splits=3)
    subsets = subset_assignment_summary(subset_ass, num_samples, tgt_n_items)
    subsets

    Output:
                num_items	deviation
    [4, 5]	    32.0	3.666667
    [3, 2, 6]	28.0	-0.333333
    [1, 0]	    25.0	-3.333333
    ```

    """
    assert input_cost_per_item.ndim == 1, "input_cost_per_item must be a 1D array"
    n_items = input_cost_per_item.shape[0]
    # assert n_items % n_splits == 0, 'n_splits must be a divisor of the number of items'

    assignment = defaultdict(list)
    total_cost = input_cost_per_item.sum()
    target_n_items_per_split = total_cost / n_splits

    def compute_cost_matrix(current_assignment: Dict[int, List[int]], cost_per_item: np.ndarray) -> np.ndarray:
        # compute the cost of current assignment
        # current_assignment_cost is a 1d column vector -> shape (n_splits, 1) splits are the rows
        if len(current_assignment) == 0:
            current_assignment_cost = np.zeros(n_splits)[:, None]
        else:
            current_assignment_cost = np.array(
                [np.sum(input_cost_per_item[assignment]) for assignment in current_assignment.values()]
            )[:, None]

        # compute the cost of assigning each item to each split
        cost_matrix = current_assignment_cost + cost_per_item - target_n_items_per_split
        cost_matrix = cost_func(cost_matrix)
        return cost_matrix

    cost_per_item = copy.deepcopy(input_cost_per_item)
    original_idx = np.arange(n_items)  # keep track of original index of items
    # solve a sequence of linear sum assignment problems
    while len(cost_per_item) > 0:
        # print(cost_per_item)
        cost_matrix = compute_cost_matrix(assignment, cost_per_item)
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            assignment[row].append(original_idx[col])
        cost_per_item = np.delete(cost_per_item, col_ind)
        original_idx = np.delete(original_idx, col_ind)

    # compute total assignment cost
    current_assignment_cost = {idx: np.sum(input_cost_per_item[assignment]) for idx, assignment in assignment.items()}
    # target_deviation_per_split = current_assignment_cost - target_samples_per_split
    return assignment, current_assignment_cost, target_n_items_per_split


def subset_assignment_summary(
    subset_assignment: Dict[int, List[int]],
    num_items_per_assignment: Dict[int, Union[int, float]],
    target_num_items: int,
) -> pd.DataFrame:
    """Summarize the assignment results in a single dataframe.

    Args:
        subset_assignment (Dict[int, List[int]]): Assignment of items to subsets.
        num_items_per_assignment (Dict[int, Union[int, float]]): number of items per subset
        target_num_items (int): goal number of items per subset

    Returns:
        pd.DataFrame: the summary dataframe
    """
    num_items = pd.Series(num_items_per_assignment).values
    deviation = num_items - target_num_items
    res = np.stack([num_items, deviation]).transpose()
    subset_assignments = pd.Series(subset_assignment).values
    res_df = pd.DataFrame(data=res, columns=["num_items", "deviation"], index=subset_assignments)
    return res_df


def assignment_summary_to_partition_dict(assignment_summary: pd.DataFrame) -> Dict[int, List[int]]:
    """Convert a summary dataframe to a partition dictionary.
    The partition dictionary contains the partition idx as key and the list of items as value.

    Args:
        assignment_summary (pd.DataFrame): summary dataframe

    Returns:
        Dict[int, List[int]]: partition dictionary
    """
    return {idx: assignment_summary.index[idx] for idx in range(len(assignment_summary))}
