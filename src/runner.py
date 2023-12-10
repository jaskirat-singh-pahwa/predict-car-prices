import pandas as pd
import yaml
from yaml.loader import SafeLoader
from typing import List

from logger import get_logger
from data_cleaning import (
    load_csv_data,
    get_data_without_null_values,
    get_data_without_duplicate_records,
    get_data_without_outliers,
    get_age_of_car,
    get_last_10_years_old_cars,
    # get_cleaned_data

)
# from modelling import train_models
from predictions import get_predicted_output
from process_user_input_data import get_processed_user_input_data


logger = get_logger(module_name="data-cleaning")


def parse_config_file(file_path: str):
    with open(file_path, "r") as f:
        config_data = list(yaml.load_all(f, Loader=SafeLoader))
        logger.info(f"\n\n{config_data}")
        return config_data


def run_app(user_input_file_path: str, config_file_path=None) -> pd.DataFrame:
    # config_data = parse_config_file(file_path=config_file_path)

    selected_columns: List[str] = [
        "seller",
        "offerType",
        "price",
        "abtest",
        "vehicleType",
        "yearOfRegistration",
        "gearbox",
        "powerPS",
        "model",
        "kilometer",
        "fuelType",
        "brand",
        "notRepairedDamage",
        "postalCode",
        "lastSeen"
    ]
    raw_user_data: pd.DataFrame = load_csv_data(
        input_path=user_input_file_path,
        columns=selected_columns
    )

    raw_user_data = get_data_without_null_values(df=raw_user_data)
    raw_user_data = get_data_without_duplicate_records(df=raw_user_data)
    raw_user_data = get_data_without_outliers(
        df=raw_user_data,
        columns=["price", "kilometer", "powerPS"]
    )
    raw_user_data = get_age_of_car(df=raw_user_data)
    raw_user_data = get_last_10_years_old_cars(df=raw_user_data)

    cleaned_user_data: pd.DataFrame = get_processed_user_input_data(file_path=user_input_file_path)

    predictions = get_predicted_output(
        raw_user_data=raw_user_data,
        processed_user_data=cleaned_user_data
    )
    print(predictions.head())

    # train: bool = config_data[0]["car_prices"]["run_type"]["train"]
    # predict_user_data: bool = config_data[0]["car_prices"]["run_type"]["predict"]["user_given_data_test"]
    #
    # if train:
    #     raw_used_cars_data_path: str = config_data[0]["train_car_prices"]["full_data_path_raw"]
    #     logger.info(f"\n\nRaw data path: {raw_used_cars_data_path}")
    #
    #     cleaned_data: pd.DataFrame = get_cleaned_data(data_path=raw_used_cars_data_path)
    #     logger.info(cleaned_data.shape)
    #     cleaned_data.to_csv("input/processed/modelling_data/processed_data_for_modelling.csv")
    #
    #     train_models(df=cleaned_data)
    #
    # if predict_user_data:
    #     user_file_path: str = config_data[0]["predict_car_prices"]["user_given_data_path_raw"]
    #     cleaned_user_data: pd.DataFrame = get_processed_user_input_data(file_path=user_file_path)
    #     print(cleaned_user_data.shape)

    return predictions
