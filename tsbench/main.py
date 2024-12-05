# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

from .config import Config
from .data import get_datasetgenerator
from .ml_utils.torch_utils import set_seed
from .models import adapt_model_cfg_to_datasetgenerator, get_model
from .trainer import Trainer


def main(cfg: Config):
    # set seed for reproducibility (as first thing)
    set_seed(cfg.experiment_data.seed)

    # create dataset and dataloader
    datasetgenerator = get_datasetgenerator(cfg.data)
    datasetgenerator.generate_dataset()
    cfg.model = adapt_model_cfg_to_datasetgenerator(datasetgenerator, cfg.model)

    # create model
    model = get_model(cfg.model)
    model.reset_parameters()

    # create trainer and run training
    trainer = Trainer(config=cfg, model=model, datasetgenerator=datasetgenerator)
    trainer.run()
