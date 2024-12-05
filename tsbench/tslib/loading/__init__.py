from ...ml_utils.meta_factory import get_and_create_class_factory
from .csv_loader import CSVTimeSeriesDataset, MultiCSVTimeSeriesDataset

# from ._interior_dataset import InteriorSessionDataset, InteriorSnippetDataset

# _raw_dataset_registry = {"interior_session": InteriorSessionDataset, "interior_snippet": InteriorSnippetDataset}

_raw_dataset_registry = {"csvloader": CSVTimeSeriesDataset, "multicsvloader": MultiCSVTimeSeriesDataset}

get_raw_dataset, create_raw_dataset = get_and_create_class_factory(_raw_dataset_registry)
