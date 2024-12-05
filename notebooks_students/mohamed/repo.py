import sys
from pathlib import Path

sys.path.append("../..")
from tsbench.ml_utils.output_loader.repo import Repo

REPO = Repo(dir=Path("../../"))
