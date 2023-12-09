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

    # input_type: str = ""  # Take input from UI -> file input or self input
    # logger.info(f"\n\nInput type entered by user: {input_type}")

    # if input_type.lower() == "file_input":
    #     file_input_path: str = ""  # Take file input path from UI
    #
    # elif input_type.lower() == "self_input":
    #     self_input_params = {
    #         "": ""
    #     }
    #
    # else:
    #     raise Exception("\n\nSorry! Incorrect option is chosen. Available choices ('file_input', 'self_input')!")

    user_input_file_path: str = "" # Take input from UI
    run_app(config_file_path=config_file_path, user_input_file_path=user_input_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
