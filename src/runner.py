import pandas as pd
import yaml
from yaml.loader import SafeLoader

from logger import get_logger
from data_cleaning import get_cleaned_data


# pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="data-cleaning")


def parse_config_file(file_path: str):
    with open(file_path, "r") as f:
        config_data = list(yaml.load_all(f, Loader=SafeLoader))
        logger.info(config_data)
        return config_data


def run_app(config_file_path: str):
    config_data = parse_config_file(file_path=config_file_path)

    train: str = config_data[0]["car_prices"]["run_type"]["train"]

    if train:
        raw_used_cars_data_path: str = config_data[0]["train_car_prices"]["full_data_path_raw"]
        logger.info(f"Raw data path: {raw_used_cars_data_path}")

        cleaned_data: pd.DataFrame = get_cleaned_data(data_path=raw_used_cars_data_path)
