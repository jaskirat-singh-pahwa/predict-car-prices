import sys
import pandas as pd

from typing import Dict

from args import parse_args
from logger import get_logger
from runner import run_app

pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="main")


def main(argv) -> None:
    args: Dict[str, str] = parse_args(argv)
    config_file_path: str = args["config_file"]
    logger.info(f"Config file path given: {config_file_path}")

    run_app(config_file=config_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
