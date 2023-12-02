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
        return pd.read_csv(input_path)


def get_cleaned_data(data_path: str) -> pd.DataFrame:
    used_cars: pd.DataFrame = load_csv_data(input_path=data_path)
    logger.info(used_cars.info())

    return used_cars
