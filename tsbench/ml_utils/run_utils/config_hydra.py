# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Korbinian Poeppel

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Union

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


def config_yaml_to_cmdline(config_yaml: str, override: str = "") -> List[str]:
    # override either "", "+", "++" see hydra
    cfg = OmegaConf.create(config_yaml)
    cfg_dict = OmegaConf.to_container(cfg)
    cmdline_opts = []

    def dict_to_cmdlines(dct: Union[Dict, List, str, int, float], prefix: str = ""):
        cmdlines = []

        if isinstance(dct, Dict):
            for sub_cfg in dct:
                newprefix = (prefix + "." if prefix else "") + sub_cfg
                cmdlines += dict_to_cmdlines(dct[sub_cfg], prefix=newprefix)
        elif isinstance(dct, List):
            cmdlines.append(override + prefix + "=[" + ",".join(map(str, range(len(dct)))) + "]")
            for n, sub_cfg in enumerate(dct):
                cmdlines += dict_to_cmdlines(sub_cfg, prefix=(prefix + "." if prefix else "") + str(n))
        else:
            cmdlines.append(override + prefix + "=" + str(dct))
        return cmdlines

    # old version
    # for sub_cfg in cfg_dict:
    #     print(sub_cfg, cfg_dict[sub_cfg])
    #     cfg_json = json.dumps(cfg_dict[sub_cfg], separators=(',', ':'))
    #     cmdline_opts.append(override + sub_cfg + "=" + cfg_json)
    cmdline_opts = dict_to_cmdlines(cfg_dict, prefix="")
    # print(cmdline_opts)
    return cmdline_opts


def run_hydra(
    config_path: str = "./config",
    config_name: str = "default",
    cmdline_opts=[],
    config_yaml: str = "",
    config_yaml_override_opt: str = "++",
):
    # do not actually run hydra as a separate executable

    config_path = config_path if os.path.isabs(config_path) else os.path.abspath(config_path)
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(
            config_name=config_name,
            overrides=cmdline_opts + config_yaml_to_cmdline(config_yaml, override=config_yaml_override_opt),
        )

    return OmegaConf.to_yaml(cfg)


if __name__ == "__main__":
    main()
