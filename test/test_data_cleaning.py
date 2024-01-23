import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_cleaning import (
    get_data_without_duplicate_records,
    get_data_without_null_values
)


def test_get_data_without_null_values() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_null_records.csv")
    actual_result: pd.DataFrame = get_data_without_null_values(
        df=sample_df
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_without_null_records.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True)
    )


def test_get_data_without_duplicate_records() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_duplicate_records.csv")
    actual_result: pd.DataFrame = get_data_without_duplicate_records(
        df=sample_df
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_without_duplicates.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True)
    )

