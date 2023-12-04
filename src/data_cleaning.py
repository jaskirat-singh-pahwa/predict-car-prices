import pandas as pd
from typing import Optional, List

from logger import get_logger


pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="data-cleaning")


def load_csv_data(input_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is not None:
        return pd.read_csv(
            filepath_or_buffer=input_path,
            header=0,
            usecols=columns
        )
    else:
        return pd.read_csv(filepath_or_buffer=input_path, encoding="ISO-8859-1")


def get_data_without_null_values(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_data_without_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_data_with_resolved_schemas(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_data_without_outliers(df: pd.DataFrame) -> pd.DataFrame:
    pass


def replace_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_age_of_car(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_last_10_years_old_cars(df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_german_to_english_features() -> pd.DataFrame:
    pass


def get_scaled_data() -> pd.DataFrame:
    pass


def get_encoded_data() -> pd.DataFrame:
    pass


def get_cleaned_data(data_path: str) -> pd.DataFrame:
    used_cars: pd.DataFrame = load_csv_data(input_path=data_path)
    logger.info(used_cars.info())

    return used_cars
