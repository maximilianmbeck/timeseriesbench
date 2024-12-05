# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Type, Union

import pandas as pd
from s3pathlib import S3Path
from tqdm import tqdm

from ..base.timeseries_dataset import TimeSeries, TimeSeriesDataset, TimeSeriesMeta
from ..consts.dataset_columns import DC
from .dataset_folder import DatasetFolderS3Path

LOGGER = logging.getLogger(__name__)

###! THIS FILE IS FOR REFERENCE ONLY
#! DO NOT USE!


@dataclass
class DrivingTimeSeriesMeta(TimeSeriesMeta):
    vehicle_id: str


class DrivingTimeSeries(TimeSeries):
    def __init__(self, index: int, key: str, dataframe: pd.DataFrame):
        self._dataframe = dataframe
        self._index = index
        self._key = key
        # create meta data upon construction as the dataframe might change
        # during lifetime of this object due to transformations
        self._vehicle_id = int(self._dataframe[DC.vehicle_id].iloc[0])
        self._meta_data = DrivingTimeSeriesMeta(
            key=self.key,
            index=self.index,
            vehicle_id=self.vehicle_id,
            features=self.features,
            num_timesteps=len(self),
        )

    @property
    def features(self) -> List[str]:
        return list(set(self.dataframe.columns.tolist()) - set([DC.datetime, DC.vehicle_id]))

    @property
    def index(self) -> int:
        return self._index

    @property
    def key(self) -> str:
        return self._key

    @property
    def vehicle_id(self) -> int:
        return self._vehicle_id

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    @property
    def meta_data(self) -> DrivingTimeSeriesMeta:
        return self._meta_data

    def duration(self, format: str = "ms") -> float:
        duration_ms = self.dataframe[DC.datetime].max() - self.dataframe[DC.datetime].min()
        if format == "ms":
            return duration_ms
        elif format == "s":
            return duration_ms / 1000
        elif format == "m":
            return duration_ms / 1000 / 60


class Snippet(DrivingTimeSeries):
    def __init__(self, snippet_idx: int, snippet_key: str, snippet_data: pd.DataFrame):
        super().__init__(index=snippet_idx, key=snippet_key, dataframe=snippet_data)

    @property
    def sampling_interval(self) -> float:
        return round(self.dataframe[DC.datetime].diff().mean(), 2)

    @property
    def features(self) -> List[str]:
        return list(set(super().features) - set([DC.snippet_id]))


class Session(DrivingTimeSeries):
    """Wrapper for a driving session partitioned in snippets."""

    def __init__(self, session_idx: int, session_key: str, session_data: pd.DataFrame):
        super().__init__(index=session_idx, key=session_key, dataframe=session_data)
        self._snippet_ids = None
        self._snippet_count = None

    def __getitem__(self, key: Union[str, int]) -> Snippet:
        assert isinstance(key, (str, int)), f"Key must be either a string or an integer, got {type(key)}"
        if isinstance(key, int):
            idx = key
            key = self.snippet_ids[key]
        elif isinstance(key, str):
            idx = self.snippet_ids.index(key)

        snippet_data = self.dataframe.loc[self.dataframe[DC.snippet_id] == key]

        return Snippet(snippet_idx=idx, snippet_key=key, snippet_data=snippet_data)

    @property
    def snippet_ids(self) -> List[str]:
        # lazy evaluate
        if self._snippet_ids is None:
            self._snippet_ids = self.dataframe[DC.snippet_id].unique()
        return self._snippet_ids.tolist()

    @property
    def snippet_count(self) -> int:
        # lazy evaluate
        if self._snippet_count is None:
            self._snippet_count = self.dataframe[DC.snippet_id].nunique()
        return self._snippet_count

    @property
    def sampling_interval(self) -> float:
        return round(self[0].dataframe[DC.datetime].diff().mean(), 2)

    @property
    def features(self) -> List[str]:
        return list(set(super().features) - set([DC.session_id, DC.snippet_id]))


class InteriorSessionDataset(TimeSeriesDataset):
    """Dataset that provides access to driving sessions."""

    def __init__(self, folder: Union[str, S3Path], **kwargs):
        self._dataset = DatasetFolderS3Path(folder, **kwargs)

    def __getitem__(self, key: Union[str, int]) -> Session:
        assert isinstance(key, (str, int)), f"Key must be either a string or an integer, got {type(key)}"

        session_data = self._dataset[key]

        if isinstance(key, int):
            idx = key
            key = self._dataset.idx_to_key[key]
        elif isinstance(key, str):
            idx = self._dataset.key_to_idx[key]

        return Session(session_idx=idx, session_key=key, session_data=session_data)

    def __len__(self):
        return len(self._dataset)

    @property
    def root(self) -> S3Path:
        return self._dataset.folder


class InteriorSnippetDataset(TimeSeriesDataset):
    """Dataset that provides access to driving snippets."""

    def __init__(
        self, folder: Union[str, S3Path], index_file: str = "snippet_idx.p", rebuild_index: bool = False, **kwargs
    ):
        # cache by default
        if not "cache_data" in kwargs:
            kwargs["cache_data"] = True
        self._dataset = InteriorSessionDataset(folder, **kwargs)
        self._index_file = index_file
        self._rebuild_index = rebuild_index

        # index for snippets
        self._dataset._dataset.cache_data = False  # disable caching for indexing
        # tuple contains session key and snippet key
        self._index = {}
        self._snippet_idx_to_keys: Dict[int, Tuple[str, str]] = {}
        self._snippet_key_to_keys: Dict[str, Tuple[str, str]] = {}
        self._snippet_key_to_ts_meta: Dict[str, Dict[str, Any]] = {}
        index = self._init_index()
        self._snippet_idx_to_keys = index["snippet_idx_to_keys"]
        self._snippet_key_to_keys = index["snippet_key_to_keys"]
        self._snippet_key_to_ts_meta = index["snippet_key_to_ts_meta"]
        self._index = index
        self._dataset._dataset.cache_data = True  # enable caching again

    @property
    def index(self) -> Dict[str, Any]:
        return self._index

    @property
    def root(self) -> S3Path:
        return self._dataset.root

    def _init_index(self) -> Tuple[Dict, Dict]:
        """Build index for snippets."""
        # try loading index from file
        index = {}
        if self._index_file and not self._rebuild_index:
            index = self._load_index(self._index_file)
        else:
            index = self._build_index()

        if index and len(index) >= 3:
            return index
        else:
            raise ValueError("Index must contain two keys: snippet_idx_to_keys and snippet_key_to_keys.")

    def _build_index(self, save_index: bool = True) -> Dict[str, Any]:
        print("Building index for snippet dataset...")
        start = time.time()
        snippet_idx_to_keys = {}
        snippet_key_to_keys = {}
        snippet_key_to_ts_meta = {}
        global_snippet_idx = 0
        for session_idx in tqdm(range(len(self._dataset)), desc="Sessions", file=sys.stdout):
            session = self._dataset[session_idx]
            for snippet_idx in range(session.snippet_count):
                snippet = session[snippet_idx]
                snippet_idx_to_keys[global_snippet_idx] = (session.key, snippet.key)
                snippet_key_to_keys[snippet.key] = (session.key, snippet.key)
                snippet_key_to_ts_meta[snippet.key] = asdict(snippet.meta_data)
                global_snippet_idx += 1
        elapsed = time.time() - start
        print(f"Index built in {elapsed:.3f} seconds.")
        # build meta data table
        print("Building meta data summary...")
        ts_meta = []
        meta_data_types: Tuple[Type] = (int, str, float)
        for key in tqdm(snippet_key_to_ts_meta, "Meta data snippet", file=sys.stdout):
            ts_meta.append({k: v for k, v in snippet_key_to_ts_meta[key].items() if isinstance(v, meta_data_types)})

        index = {
            "snippet_idx_to_keys": snippet_idx_to_keys,
            "snippet_key_to_keys": snippet_key_to_keys,
            "snippet_key_to_ts_meta": snippet_key_to_ts_meta,
            "meta_data_summary": ts_meta,
        }
        if save_index:
            self._save_index(index=index, index_file=self._index_file)
            print(f"Index saved to {self._index_file}.")
        return index

    def _load_index(self, index_file: str) -> Dict[str, Any]:
        """Load index from file."""
        index_path = self.root / index_file
        try:
            with index_path.open("rb") as f:
                index = pickle.load(f)
        except:
            index = {}
        return index

    def _save_index(self, index: Dict[str, Dict], index_file: str):
        """Save index to file."""
        index_path = self.root / index_file
        with index_path.open("wb") as f:
            pickle.dump(index, f)

    def __getitem__(self, key: Union[str, int]) -> Snippet:
        assert isinstance(key, (str, int)), f"Key must be either a string or an integer, got {type(key)}"

        if isinstance(key, int):
            session_key, snippet_key = self._snippet_idx_to_keys[key]
        elif isinstance(key, str):
            session_key, snippet_key = self._snippet_key_to_keys[key]

        session = self._dataset[session_key]
        snippet = session[snippet_key]

        return snippet

    def __len__(self):
        return len(self._snippet_idx_to_keys.keys())

    def get_meta_data_summary(
        self,
        meta_data_types: Tuple[Type] = (int, str, float),
        indices: List[int] = [],
    ) -> pd.DataFrame:
        # Do not create the meta data summary on the fly.
        # self._dataset._dataset.cache_data = False
        # meta_data_summary = super().get_meta_data_summary(meta_data_types)
        # self._dataset._dataset.cache_data = True
        # Instead, access the prebiult meta data summary.
        meta_data_summary = pd.DataFrame(self.index["meta_data_summary"])
        if indices:
            return meta_data_summary.iloc[indices]
        return meta_data_summary
