import pandas as pd

from data_cleaning import get_cleaned_data


def get_processed_user_input_data(file_path: str) -> pd.DataFrame:
    cleaned_user_data: pd.DataFrame = get_cleaned_data(data_path=file_path)
    print(cleaned_user_data)
    return cleaned_user_data
