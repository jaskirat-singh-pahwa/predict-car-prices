import pandas as pd
import yaml
from yaml.loader import SafeLoader

from logger import get_logger
from data_cleaning import get_cleaned_data
from modelling import train_models
from predictions import get_predicted_output
from process_user_input_data import get_processed_user_input_data


logger = get_logger(module_name="data-cleaning")


def parse_config_file(file_path: str):
    with open(file_path, "r") as f:
        config_data = list(yaml.load_all(f, Loader=SafeLoader))
        logger.info(f"\n\n{config_data}")
        return config_data


def run_app(user_input_file_path: str, config_file_path=None,):
    config_data = parse_config_file(file_path=config_file_path)

    user_input_file_path: str = "input/raw/user_data/raw_user_data.csv"  # Take input from UI
    # raw_user_data: pd.DataFrame = pd.read_csv(
    #     filepath_or_buffer=user_input_file_path,
    #     encoding="ISO-8859-1"
    # )

    cleaned_user_data: pd.DataFrame = get_processed_user_input_data(file_path=user_input_file_path)

    predictions = get_predicted_output(df=cleaned_user_data)
    print(predictions.head())

    train: bool = config_data[0]["car_prices"]["run_type"]["train"]
    predict_user_data: bool = config_data[0]["car_prices"]["run_type"]["predict"]["user_given_data_test"]

    if train:
        raw_used_cars_data_path: str = config_data[0]["train_car_prices"]["full_data_path_raw"]
        logger.info(f"\n\nRaw data path: {raw_used_cars_data_path}")

        cleaned_data: pd.DataFrame = get_cleaned_data(data_path=raw_used_cars_data_path)
        logger.info(cleaned_data.shape)
        cleaned_data.to_csv("input/processed/modelling_data/processed_data_for_modelling.csv")

        train_models(df=cleaned_data)

    if predict_user_data:
        user_file_path: str = config_data[0]["predict_car_prices"]["user_given_data_path_raw"]
        cleaned_user_data: pd.DataFrame = get_processed_user_input_data(file_path=user_file_path)
        print(cleaned_user_data.shape)

