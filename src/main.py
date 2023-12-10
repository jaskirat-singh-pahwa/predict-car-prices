import pandas as pd
# import sys

# from typing import Dict

# from args import parse_args
from logger import get_logger
# from runner import run_app
from streamlit_app import run_streamlit_app

pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="main")

"""
This is the main entry point of the project. From here it will call run_streamlit_app and start processing.

"""


def main() -> None:
    # args: Dict[str, str] = parse_args(argv)
    # config_file_path: str = args["config_file"]
    # logger.info(f"Config file path given: {config_file_path}")

    # run_app(config_file_path=config_file_path, user_input_file_path="")

    run_streamlit_app()


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
