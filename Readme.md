# tsbench

Hi there, this is the repository for the *tsbench* Practical Work Seminar at JKU Linz. 

The supervisors are Maximilian Beck, Oleksandra (Sasha) Prudnikova, Andreas Auer and Korbinian Pöppel.

## Setup

1. Setup a conda environment with `environment_pt2.1.0cu118` or `environment_pt2.1.0cu121`. Depending on our Nvidia GPU driver.
2. Highly recommendend: Use Microsoft Visual Studio Code for developing. 

## Quick Start Guide

In order to get started, please have a look at the tutorial notebooks in `notebooks/tutorials` and go through them in the specified order.
What you see is the process it needs to go from raw data (in this case a .csv file) to learning curves.

After you have gone over these tutorials, run your first experiments. 
For this, have a look at the folder `notebooks/experiments`. There you will find notebooks with runnable experiment configs. 
Executing the cells in this notebooks will copy these configs into the `configs` folder. You might need to create the `configs/` folder in the repository root first.

From there you can run those experiments by executing this command which will run a transformer architecture on 
a sequential cifar dataset for example:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config-name sCF10-causalselfattention--lr_0.yaml
```


## Extending tslib

### Memory Consumption and Caching

For large datasets the caching in the TargetGenerator might consume all memory available.
Some ideas to avoid this:
- `Chunk saving:` Chunk the dataset and save those chunks to disk. Keep track which samples are in which chunk and load the respective chunk on demand. 
With random access (as a dataloader with `shuffle=True` does it) this might not be very efficient as it causes alot of loading from disk. 
- `np.memmap`: As a more sophisticated and scalable solution one could use `np.memmap` (see [here](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)). This us used for LLM pretraining datasets and should be therefore scale to very large datasets (example: [EleutherAI&fairseq](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/data/indexed_dataset.py)).
- More ideas: For more caching ideas one could potentially also have a closer look at huggingface datasets how they solved this. They have similar problems with tokenization.

### Processing speed

Andi Auer profiled an extended version of tslib (more transformations implemented): 

```
[16:04:44] [INFO] [dku.utils]  -    1178.89 partition_getitem
[16:04:44] [INFO] [dku.utils]  -    761.31 partition_getitem.S3Folder_getitem
[16:04:44] [INFO] [dku.utils]  -     20.82 partition_getitem.session_getitem
[16:04:44] [INFO] [dku.utils]  -    389.20 partition_getitem.snippet_getitem
[16:04:44] [INFO] [dku.utils]  -    162.96 FeatureSelector
[16:04:44] [INFO] [dku.utils]  -    864.48 Normalizer
[16:04:44] [INFO] [dku.utils]  -   4685.63 MissingValueHandler
[16:04:44] [INFO] [dku.utils]  -    868.90 OneHotEncoding
[16:04:44] [INFO] [dku.utils]  -     26.49 window_getitem
[16:04:45] [INFO] [dku.utils]  -    438.49 target_getitem
[16:04:45] [INFO] [dku.utils]  -     11.54 target_getitem.get_window_past_frame
[16:04:45] [INFO] [dku.utils]  -     21.74 target_getitem.get_window_frame
[16:04:45] [INFO] [dku.utils]  -      5.86 fwd_val
```

These are accumulated times. Last row (~6s) is the forward pass. We see that the preprocessing consumes a lot of time during training. So this should be probably done before the training.

In order to speed up the pipeline one could try to implement "fast" versions of some transformations with polars (see tutorial [here](https://towardsdatascience.com/pandas-dataframe-but-much-faster-f475d6be4cd4)).