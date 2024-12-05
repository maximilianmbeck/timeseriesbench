# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Markus Spanring, Maximilian Beck


import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class bc:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_rank_0(message, from_state=False):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            LOGGER.info(message)
    elif not from_state:
        LOGGER.info(message)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, outpath):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=f"Downloading {url}") as t:
        urllib.request.urlretrieve(url, filename=outpath, reporthook=t.update_to)


def download_and_unzip(url: str, outpath: Path):
    zip_filename = outpath / url.split("/")[-1]
    if not zip_filename.exists():
        download_url(url, zip_filename)
        LOGGER.info("Extracting Data")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(outpath)
    else:
        LOGGER.info(f"Found {zip_filename}. Skip download and unpacking")


def download_and_unpack(url: str, outpath: Path) -> None:
    tar_filename = outpath / url.split("/")[-1]
    if not tar_filename.exists():
        download_url(url, tar_filename)
        LOGGER.info("Extracting Data")
        shutil.unpack_archive(filename=tar_filename, extract_dir=outpath)
    else:
        LOGGER.info(f"Found {tar_filename}. Skip download and unpacking")


def download_and_untar(url: str, outpath: Path, overwrite_existing: bool = False) -> Path:
    """Download and unpack tar file to outpath. Return path to unpacked directory.

    If the download file already exists, skip download.
    If the unpacked directory already exists, skip unpacking.
    If overwrite_existing is True, overwrite existing files.
    """
    download_filename = outpath / url.split("/")[-1]
    if not download_filename.exists() or overwrite_existing:
        LOGGER.info(f"Downloading data: {download_filename}")
        download_url(url, download_filename)
    else:
        LOGGER.info(f"Found {download_filename}. Skip download and unpacking")
    extract_dir = outpath / download_filename.stem
    if not extract_dir.exists() or overwrite_existing:
        LOGGER.info(f"Extracting Data in directory: {outpath}")
        file = tarfile.open(download_filename)
        file.extractall(outpath)
        file.close()
    else:
        LOGGER.info(f"Found {extract_dir}. Skip download and unpacking")
    return extract_dir
