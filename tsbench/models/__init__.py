import copy
from typing import Dict, Type, Union

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from ..data.basedataset import BaseSequenceDatasetGenerator
from ..ml_utils.config import NameAndKwargs
from .base import BaseSequenceModelTrain
from .lstm_multilayer import LMLSTMMultiLayer, SequenceLSTMMultiLayer
from .transformer.lm_transformer import LMTransformer
from .transformer.sequence_transformer import SequenceTransformer

_model_registry = {
    "sequence_transformer": SequenceTransformer,
    "lm_transformer": LMTransformer,
    "lm_lstmmultilayer": LMLSTMMultiLayer,
    "sequence_lstmmultilayer": SequenceLSTMMultiLayer,
}


def get_model_class(name: str) -> Type[BaseSequenceModelTrain]:
    if name in _model_registry:
        return _model_registry[name]
    else:
        assert False, f'Unknown model name "{name}". Available models are:' f" {str(_model_registry.keys())}"


def get_model(config: Union[Dict, DictConfig, NameAndKwargs]) -> BaseSequenceModelTrain:
    if not isinstance(config, NameAndKwargs):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)
        cfg = from_dict(data_class=NameAndKwargs, data=config, config=Config(strict=True))
    else:
        cfg = config
    model_class = get_model_class(cfg.name)

    model_cfg = from_dict(data_class=model_class.config_class, data=cfg.kwargs, config=Config(strict=True))

    model = model_class(model_cfg)

    return model


def adapt_model_cfg_to_datasetgenerator(
    datasetgenerator: BaseSequenceDatasetGenerator, model_cfg: NameAndKwargs
) -> NameAndKwargs:
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.kwargs["vocab_size"] = datasetgenerator.vocab_size
    model_cfg.kwargs["context_length"] = datasetgenerator.context_length
    model_cfg.kwargs["input_dim"] = datasetgenerator.input_dim
    model_cfg.kwargs["output_dim"] = datasetgenerator.output_dim
    return model_cfg
