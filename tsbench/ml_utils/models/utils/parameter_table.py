# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Markus Spanring

from copy import deepcopy
from typing import List, Tuple


def count_parameters(model) -> List[Tuple[List[str], float]]:
    """Read all named parameters and store them in a List with the respective parameter count.
    The named parameters are split into the respective submodules
    ("block.ff.weight" -> ["block", "ff", "weight"]).
    In case there are similaraly named parameters per layer e.g. 'block.0.ln'
    remove the digit. This way we can accumulate the parameters over all layers with
    similar names.
    """
    param_counts = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            s = [l for l in name.split(".") if not l.isdigit()]
            param_counts.append((s, num_params))

    return param_counts


def accumulate_counts(param_counts) -> dict:
    """Accumulate the parametes of equally named modules over all layers e.g.
    block.0.ln.weight + block.1.ln.weight + ...
    The resulting dictionary is nested in terms of depth (the digit of the blocks has
    been removed in the creation of the param_counts list).
    The number of occourences is refelected in the 'count' node.
    Example:
    [
        ("block.ln.weight", 2)
        ("block.ln.bias", 2)
        ("block.sequence_mix.weight", 5)
        ("block.sequence_mix.bias", 5)
        ("block.ln.weight", 2)
        ("block.ln.bias", 2)
        ("block.sequence_mix.weight", 5)
        ("block.sequence_mix.bias", 5)
    ]
    ->
    {
        "block":{
            "ln":{
                "weight": {
                    "count": 2,
                    "nparams": 4
                },
                "bias": {
                    "count": 2,
                    "nparams": 4
                }
            },
            "sequence_mix":{
                "weight": {
                    "count": 2,
                    "nparams": 10
                },
                "bias": {
                    "count": 2,
                    "nparams": 10
                }
            }
        }
    }
    """
    nested_dict = {"total_counts": 0, "layers": {}}
    for layer, n in param_counts:
        nested_dict["total_counts"] += n
        current_dict = nested_dict["layers"]
        for i, key in enumerate(layer):
            leaf = key

            if not leaf in current_dict:
                current_dict[leaf] = {}
                if i == len(layer) - 1:
                    current_dict[leaf]["nparams"] = 0
                    current_dict[leaf]["count"] = 0
            if i == len(layer) - 1:
                current_dict[leaf]["nparams"] += n
                current_dict[leaf]["count"] += 1
            current_dict = current_dict[leaf]

    return deepcopy(nested_dict)


def get_table_data(ndict, table, parent, total_params=None) -> List[Tuple[str, int, int, int, float]]:
    """Recursively go through the nested dictionary. If a module has weight and bias parameters add them,
    otherwise put the number of parameters for the respective module in the table.
    """
    if "count" in ndict and "nparams" in ndict:
        count = ndict["count"]
        nparams = ndict["nparams"]
        perlayer = nparams // count
        frac = 100 * (nparams / total_params)
        table.append((".".join(parent), count, perlayer, nparams, frac))
    elif "weight" in ndict and "bias" in ndict:
        count = ndict["weight"]["count"]
        nparams = ndict["bias"]["nparams"] + ndict["weight"]["nparams"]
        perlayer = nparams // count
        frac = 100 * (nparams / total_params)
        table.append((".".join(parent), count, perlayer, nparams, frac))
    else:
        for k in ndict:
            table = get_table_data(ndict[k], table, parent + [k], total_params)

    return table


def create_table(nested_counts) -> List[str]:
    """Convenience function to get a nicely formatted table."""
    table = []
    table.append(f"{'Name':<70}{'Occurrence':<15}{'Per Layer':<15}{'NParams':<15}{'Fraction':<15}")
    table.append("=" * 130)
    perlayer = 0
    total = 0
    for l, c, pl, np, f in get_table_data(
        nested_counts["layers"], table=[], parent=[], total_params=nested_counts["total_counts"]
    ):
        table.append(f"{l:<70}{c:<15}{pl:<15}{np:<15}{f:3.2f}")
        perlayer += pl
        total += np
    table.append("=" * 130)
    table.append(f"{' ':<85}{perlayer:<15}{total:<15}")
    table.append("=" * 130)

    return table


def model_parameter_table(model) -> List[str]:
    """Return the parameter table for the given model"""
    param_counts = count_parameters(model)
    nested_counts = accumulate_counts(param_counts)
    return create_table(nested_counts)
