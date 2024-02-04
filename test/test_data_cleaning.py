import pandas as pd
from pandas.testing import assert_frame_equal

from src.data_cleaning import (
    get_data_without_duplicate_records,
    get_data_without_null_values,
    get_data_with_resolved_schemas,
    get_data_without_outliers,
    replace_feature_values,
    get_age_of_car,
    get_last_10_years_old_cars,
    get_german_to_english_features,
    get_scaled_data,
    get_encoded_data,
    get_cleaned_data
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


def test_get_data_with_resolved_schemas() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_unresolved_schema_records.csv")
    actual_result: pd.DataFrame = get_data_with_resolved_schemas(
        df=sample_df
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_resolved_schema_records.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True)
    )


def test_get_data_without_outliers() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_outliers_records.csv")
    actual_result: pd.DataFrame = get_data_without_outliers(
        df=sample_df,
        columns=["price"]
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_without_outliers_records.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True)
    )


def test_replace_feature_values() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_replace_values_records.csv")
    actual_result: pd.DataFrame = replace_feature_values(
        df=sample_df
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_replace_values_records.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True)
    )


def test_get_age_of_car() -> None:
    sample_df: pd.DataFrame = pd.read_csv("test/data_cleaning/sample_records_without_age.csv")
    actual_result: pd.DataFrame = get_age_of_car(
        df=sample_df
    )

    expected_result: pd.DataFrame = pd.read_csv("test/data_cleaning/expected_records_with_age.csv")

    assert_frame_equal(
        actual_result.reset_index(drop=True),
        expected_result.reset_index(drop=True),
        check_dtype=False
    )


def test_get_last_10_years_old_cars() -> None:
    pass


def test_get_german_to_english_features() -> None:
    pass


def test_get_scaled_data() -> None:
    pass


def test_get_encoded_data() -> None:
    pass


def test_get_cleaned_data() -> None:
    pass
