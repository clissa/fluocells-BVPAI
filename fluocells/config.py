"""
This module contains project-wide settings and paths to data sources.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-05-31
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()  # type: ignore

sys.path.append(str(FLUOCELLS_PATH))
import fluocells as fluo


REPO_PATH = Path(fluo.__path__[0]).parent

# metadata
METADATA = dict(
    dataset_name="Fluorescent Neuronal Cells dataset",
    data_url="https://amsacta.unibo.it/id/eprint/7347 ",
    contributors=["Luca Clissa", "Roberto Morelli", "Antonio Macaluso", "et al."],
    current_version="v2",
)

# reproducibility
TEST_PCT = 0.25
TRAINVAL_TEST_SEED = 10

# data
RAW_DATA_PATH = REPO_PATH / "raw_data"

# EDIT HERE TO ROOT PATH OF THE DATA FOLDER
DATA_PATH = (
    REPO_PATH.parent
    / "fluocells-scientific-data"
    / f"dataset_{METADATA['current_version']}"
)

DATA_PATH_r = DATA_PATH / "red"
DATA_PATH_y = DATA_PATH / "yellow"
DATA_PATH_g = DATA_PATH / "green"

UNLABELLED_IMG_PATH_r = DATA_PATH_r / "unlabelled/images"
UNLABELLED_IMG_PATH_y = DATA_PATH_y / "unlabelled/images"
UNLABELLED_IMG_PATH_g = DATA_PATH_g / "unlabelled/images"

DEBUG_PATH = REPO_PATH / "debug"

# models
MODELS_PATH = REPO_PATH / "models"
