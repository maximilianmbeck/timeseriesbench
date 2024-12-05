# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import math

import numpy as np


# bitreversal permutation is a copy from:
# https://github.com/HazyResearch/safari/blob/26f6223254c241ec3418a0360a3a704e0a24d73d/src/utils/permutations.py#L8
def bitreversal_po2(n):
    m = int(math.log(n) / math.log(2))
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)


def bitreversal_permutation(n):
    m = int(math.ceil(math.log(n) / math.log(2)))
    N = 1 << m
    perm = bitreversal_po2(N)
    return np.extract(perm < n, perm)
