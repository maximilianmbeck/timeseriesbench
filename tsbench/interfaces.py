# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence


class SequenceInterface(ABC):
    """This is a generic interface for a sequence.
    In our case a sequence also includes its label. Therefore, the label (aka output_dim)
    is also part of the sequence interface.
    A sequence always has a length (=context_length).

    A sequence can have one of the following flavors:
    Input sequence:
    - sequence of tokens (e.g. words): vocab_size must be specified
    - sequence of vectors: input_dim must be specified

    Output sequence:
    - next token (e.g. word): `vocab_size` must be specified (e.g. Causal Language Modeling).
    - (sequence of) vectors: output_dim must be specified (e.g. Forecasting)
    - label: output_dim must be specified (e.g. Sequence Classification)

    Examples:
    - Causal Language Modeling: input_dim = None, output_dim = None, vocab_size = int
    - Forecasting: input_dim = int, output_dim = int, vocab_size = None
    - Sequence Classification (General Sequence): input_dim = int, output_dim = int, vocab_size = None
    - Sequence Classification (Text): input_dim = None, output_dim = int, vocab_size = int
    """

    @property
    def input_dim(self) -> Optional[Sequence[int]]:
        return None

    @property
    def output_dim(self) -> Optional[Sequence[int]]:
        return None

    @property
    def vocab_size(self) -> Optional[int]:
        return None

    @property
    @abstractmethod
    def context_length(self) -> int:
        pass


@dataclass
class SequenceConfigInterface:
    context_length: int
    vocab_size: Optional[int] = None
    input_dim: Optional[Sequence[int]] = None
    output_dim: Optional[Sequence[int]] = None


class Tokenizer(Protocol):
    def __call__(self, **kwargs) -> Any:
        ...

    def __len__(self) -> int:
        ...


class TokenizerInterface(ABC):
    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        return None
