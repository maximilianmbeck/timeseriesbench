# Copyright 2023 JKU Linz, All Rights Reserved
# Author: Maximilian Beck


import logging
import shutil
import sys
from pathlib import Path

from ..utils import download_and_untar

LOGGER = logging.getLogger(__name__)
LRA_DOWNLOAD_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"

LRA_DOWNLOAD_FOLDER = "lra_download"
LRA_PREPARED_FOLDER = "lra_prepared"


def download_and_prepare_lra_data(
    data_dir: Path, delete_downloaded: bool = False, overwrite_existing: bool = False
) -> None:
    data_dir.mkdir(exist_ok=True)

    lra_download = data_dir / LRA_DOWNLOAD_FOLDER
    lra_download.mkdir(exist_ok=True)

    lra_extracted = download_and_untar(
        url=LRA_DOWNLOAD_URL, outpath=lra_download, overwrite_existing=overwrite_existing
    )

    # create new lra directory
    lra_prepared = data_dir / LRA_PREPARED_FOLDER
    lra_prepared.mkdir(exist_ok=True)

    def _copy_lra_dir(src_dir: str, dest_dir: str):
        src = lra_extracted / src_dir
        dest = lra_prepared / dest_dir
        if not dest.exists() or overwrite_existing:
            LOGGER.info(f"Copying {src_dir}..")
            shutil.copytree(src=src, dst=dest, dirs_exist_ok=overwrite_existing)
        else:
            LOGGER.info(f"Found {dest}. Skip copying {src_dir}")

    # copy files from /listops-1000/ to lra_prepared/listops/
    _copy_lra_dir(src_dir="lra_release/listops-1000", dest_dir="listops")

    # copy files from tsv_data/ to lra_prepared/aan/
    _copy_lra_dir(src_dir="lra_release/tsv_data", dest_dir="aan")

    # copy files /pathfinderXX/ to lra_prepared/pathfinder/
    _copy_lra_dir(src_dir="lra_release/pathfinder32", dest_dir="pathfinder/pathfinder32")
    _copy_lra_dir(src_dir="lra_release/pathfinder64", dest_dir="pathfinder/pathfinder64")
    _copy_lra_dir(src_dir="lra_release/pathfinder128", dest_dir="pathfinder/pathfinder128")
    _copy_lra_dir(src_dir="lra_release/pathfinder256", dest_dir="pathfinder/pathfinder256")

    if delete_downloaded:
        LOGGER.info(f"Deleting downloaded files: {lra_download}")
        shutil.rmtree(lra_download)


if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], encoding="utf-8", level=logging.DEBUG)

    download_and_prepare_lra_data(data_dir=Path("./data/test_lra"))
