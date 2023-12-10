import pandas as pd

from data_cleaning import get_cleaned_data


"""
This module is for processing user input data and making it compatible with the data on which 
the training has been done.

"""


def get_processed_user_input_data(file_path: str) -> pd.DataFrame:
    cleaned_user_data: pd.DataFrame = get_cleaned_data(data_path=file_path)

    processed_modelling_data: pd.DataFrame = pd.read_csv(
        filepath_or_buffer="input/processed/modelling_data/processed_data_for_modelling.csv",
        encoding="ISO-8859-1"
    )
    # columns_not_in_processed_data = cleaned_user_data.columns.difference(processed_modelling_data.columns)
    # print("-----------", len(columns_not_in_processed_data))

    missing_columns = set(processed_modelling_data.columns) - set(cleaned_user_data.columns)
    print(len(missing_columns))

    for col in missing_columns:
        cleaned_user_data[col] = 0  # or any default value you want

    feature_names = processed_modelling_data.columns.tolist()
    cleaned_user_data = cleaned_user_data[feature_names]

    return cleaned_user_data
