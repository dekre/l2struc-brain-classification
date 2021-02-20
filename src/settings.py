from pathlib import Path

DATA_BASE_DIR = Path(__file__).parent.parent.joinpath("data")
DATA_SRC_DIR = Path(DATA_BASE_DIR).joinpath("in")
DATA_DIS_DIR = Path(DATA_BASE_DIR).joinpath("out")

DATA_SRC_ZIP = Path(DATA_SRC_DIR).joinpath("guest-20210220_031112.zip")
DATA_SRC_LBL = Path(DATA_SRC_DIR).joinpath("labels.csv")