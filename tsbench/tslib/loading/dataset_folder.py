# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck

import logging
from io import BytesIO
from typing import Callable, Dict, List, Union

# import dataiku
import pandas as pd
from s3pathlib import S3Path

LOGGER = logging.getLogger(__name__)


class DatasetFolder:
    """Provides a dictionary-like interface to a folder of datasets.
    Assumes all dataframes in the folder have the same schema.
    Assumes all dataframes are sorted by the datetime column.
    TODO add check for schema of different datasets
    TODO add check for datetime column
    """

    def __init__(
        self,
        folder: dataiku.Folder,
        cache_data: bool = True,
        from_bytes_to_df: Callable[[BytesIO], pd.DataFrame] = pd.read_parquet,
        **kwargs,
    ):
        self._folder = folder
        self._cache_data = cache_data
        self._from_bytes_to_df = from_bytes_to_df

        self._index_key_name = None
        self._idx_to_key = {}
        self._key_to_idx = {}
        self._key_to_path: Dict[str, str] = {}  # main access via key
        self._cache: Dict[str, pd.DataFrame] = {}

        self._init()

    def _init(self):
        # go through all paths in the folder and extract the keys
        LOGGER.debug("Initializing DatasetFolder.")
        for idx, path in enumerate(self._folder.list_paths_in_partition()):
            if not self._is_dataset_path(path):
                print(f"DatasetFolder: Skipping path {path}")
                continue
            if self._index_key_name is None:
                self._index_key_name = self._extract_index_key_name_for_path(path)
            else:
                assert self._index_key_name == self._extract_index_key_name_for_path(
                    path
                ), f"All paths must have the same index key name, unmatching key name found in path {path}. Should be {self._index_key_name}"

            key = self._extract_index_key_value_for_path(path)
            assert key not in self._key_to_idx, f"Duplicate key {key} in the DatasetFolder"
            self._idx_to_key[idx] = key
            self._key_to_idx[key] = idx
            self._key_to_path[key] = path
        LOGGER.debug("DatasetFolder initialized.")

    def _extract_index_key_name_for_path(self, path: str) -> str:
        return path.split("/")[1].split("=")[0]

    def _extract_index_key_value_for_path(self, path: str) -> str:
        return path.split("/")[1].split("=")[1]

    def _is_dataset_path(self, path: str) -> bool:
        """If the path contains a `=` it is assumed to be a dataset path."""
        if "__HIVE_DEFAULT_PARTITION__" in path:
            print(
                f"WARNING: DatasetFolder: Skipping path {path} (Hive default partition). The partitioning column likely contains null values."
            )
            return False
        return "=" in path and path.endswith(".parquet")

    def _load_path(self, path: str) -> pd.DataFrame:
        with self._folder.get_download_stream(path) as stream:
            bytes_file = BytesIO(stream.read())
        return self._from_bytes_to_df(bytes_file)

    @property
    def index_key_name(self):
        self._index_key_name

    @property
    def cache_data(self):
        return self._cache_data

    @cache_data.setter
    def cache_data(self, value: bool) -> None:
        self._cache_data = value

    @property
    def keys(self):
        return self._key_to_path.keys()

    @property
    def idx_to_key(self) -> Dict[int, str]:
        return self._idx_to_key

    @property
    def key_to_idx(self) -> Dict[str, int]:
        return self._key_to_idx

    def __getitem__(self, key: Union[int, str]) -> pd.DataFrame:
        assert isinstance(key, (str, int)), "key must be either a string or an integer"

        if isinstance(key, int):
            key = self._idx_to_key[key]

        if self._cache_data:
            if key not in self._cache:
                self._cache[key] = self._load_path(self._key_to_path[key])
            return self._cache[key]
        else:
            return self._load_path(self._key_to_path[key])

    def __len__(self):
        return len(self._key_to_path)


class DatasetFolderS3Path:
    """Provides a dictionary-like interface to a folder of datasets.
    Assumes all dataframes in the folder have the same schema.
    Assumes all dataframes are sorted by the datetime column.
    TODO add check for schema of different datasets
    TODO add check for datetime column
    """

    def __init__(
        self,
        folder: Union[str, S3Path],
        cache_data: bool = True,
        from_bytes_to_df: Callable[[BytesIO], pd.DataFrame] = pd.read_parquet,
        **kwargs,
    ):
        if isinstance(folder, str):
            folder = S3Path(folder)
        assert isinstance(folder, S3Path), "folder must be an S3Path object"
        assert folder.exists() and folder.is_dir(), f"folder must be a directory. Given path: {folder}"
        self.folder = folder
        self._cache_data = cache_data
        self._from_bytes_to_df = from_bytes_to_df

        self._index_key_name = None
        self._idx_to_key = {}
        self._key_to_idx = {}
        self._key_to_path: Dict[str, str] = {}  # main access via key
        self._cache: Dict[str, pd.DataFrame] = {}

        self._init()

    def _init(self):
        # go through all paths in the folder and extract the keys
        LOGGER.debug("Initializing DatasetFolder.")
        self._perform_data_dir_checks(self.folder)
        iterproxy = self.folder.iter_objects().filter(*self._get_datasetfile_filters())
        for idx, path in enumerate(iterproxy):
            if self._index_key_name is None:
                self._index_key_name = self._extract_index_key_name_for_path(path)
            else:
                assert self._index_key_name == self._extract_index_key_name_for_path(
                    path
                ), f"All paths must have the same index key name, unmatching key name found in path {path}. Should be {self._index_key_name}"

            key = self._extract_index_key_value_for_path(path)
            assert key not in self._key_to_idx, f"Duplicate key {key} in the DatasetFolder"
            self._idx_to_key[idx] = key
            self._key_to_idx[key] = idx
            self._key_to_path[key] = path
        LOGGER.debug("DatasetFolder initialized.")

    def _extract_index_key_name_for_path(self, path: S3Path) -> str:
        return path.uri.split("/")[-2].split("=")[0]

    def _extract_index_key_value_for_path(self, path: S3Path) -> str:
        return path.uri.split("/")[-2].split("=")[1]

    def _get_datasetfile_filters(self) -> List[Callable[[S3Path], bool]]:
        """Returns a list of filters that can be used to filter the dataset files."""
        return [
            lambda x: x.ext == ".parquet",
            lambda x: "=" in x.uri,
            lambda x: not "__HIVE_DEFAULT_PARTITION__" in x.uri,
        ]

    def _perform_data_dir_checks(self, folder: S3Path):
        """Performs some checks on the data directory."""
        assert len(list(folder.iter_objects())) > 0, f"Given path {folder} is empty."
        iterproxy = folder.iter_objects().filter(lambda x: "__HIVE_DEFAULT_PARTITION__" in x.uri)
        hive_default_partitions = list(iterproxy)
        if len(hive_default_partitions) > 0:
            print(
                f"WARNING: DatasetFolder: Found {len(hive_default_partitions)} Hive default partitions in the data directory. The partitioning column likely contains null values."
            )
            for path in hive_default_partitions:
                print(f"WARNING: DatasetFolder: Skipping path {path}")

    def _load_path(self, path: S3Path) -> pd.DataFrame:
        with path.open("rb") as f:
            df = self._from_bytes_to_df(f)
        return df

    @property
    def index_key_name(self):
        self._index_key_name

    @property
    def cache_data(self):
        return self._cache_data

    @cache_data.setter
    def cache_data(self, value: bool) -> None:
        self._cache_data = value

    @property
    def keys(self):
        return self._key_to_path.keys()

    @property
    def idx_to_key(self) -> Dict[int, str]:
        return self._idx_to_key

    @property
    def key_to_idx(self) -> Dict[str, int]:
        return self._key_to_idx

    def __getitem__(self, key: Union[int, str]) -> pd.DataFrame:
        assert isinstance(key, (str, int)), "key must be either a string or an integer"

        if isinstance(key, int):
            key = self._idx_to_key[key]

        if self._cache_data:
            if key not in self._cache:
                self._cache[key] = self._load_path(self._key_to_path[key])
            return self._cache[key]
        else:
            return self._load_path(self._key_to_path[key])

    def __len__(self):
        return len(self._key_to_path)
