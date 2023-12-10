import pandas as pd
from typing import Optional, List

from sklearn.preprocessing import MinMaxScaler

from logger import get_logger


pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="data-cleaning")

"""
This module is used to data cleaning
We will perform series of actions in this module on raw dataset:
    1. Feature selection - removing date crawled and other unnecessary features
    2. Handling null values
    3. Removing duplicate records
    4. Resolving schema inconsistencies
    5. Handling outliers
    6. Handling inconsistencies in features
    7. Feature Extraction -> how old is the car
    8. Filtering out all observations where car is more than 20 years old
    9. Normalizing data
    10. Check for 0 values in features and again doing Feature Selection
    11. Replacing mixture of german and english words features to english
    12. Encoding categorical values

"""


# Loading data from csv to pandas dataframe
def load_csv_data(input_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is not None:
        return pd.read_csv(
            filepath_or_buffer=input_path,
            header=0,
            usecols=columns,
            encoding="ISO-8859-1"
        )
    else:
        return pd.read_csv(filepath_or_buffer=input_path, encoding="ISO-8859-1")


# Dropping all rows if any of the columns have null value
def get_data_without_null_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


# Dropping duplicate records
def get_data_without_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


# Changing data type of price column from integer to float
def get_data_with_resolved_schemas(df: pd.DataFrame) -> pd.DataFrame:
    return df["price"].astype(float, inplace=True)


# Removing outliers from the dataframe
def get_data_without_outliers(
        df: pd.DataFrame,
        columns: List[str],
        factor=1.5
) -> pd.DataFrame:
    for column in columns:
        # Calculate the first and third quartiles
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        # Calculate the IQR (Inter quartile Range)
        iqr = q3 - q1

        # Define the lower and upper bounds for outlier detection
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Create a mask to identify outliers
        outlier_mask = ((df[column] >= lower_bound) & (df[column] <= upper_bound))

        # Filter the DataFrame to exclude outliers
        df = df[outlier_mask]

    return df


# Replacing feature values from German language to English language
def replace_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """
        Handling inconsistencies in features -
        Changing values of 2 features from German to English
            1. offerType - `Gesuch` and `Angebot` which means `request` and `listing`
            2. notRepairedDamage - "ja" and "nein" which means `yes` and `no`
    """

    def replace_values(df_with_german_values, feature_name, old_value, new_value):
        df_with_german_values[feature_name] = df_with_german_values[feature_name] \
                                                .replace(old_value, new_value)
        return df

    df = replace_values(df_with_german_values=df, feature_name="offerType", old_value="Gesuch", new_value="request")

    df = replace_values(df_with_german_values=df, feature_name="offerType", old_value="Angebot", new_value="listing")
    df = replace_values(df_with_german_values=df, feature_name="notRepairedDamage", old_value="ja", new_value="yes")
    df = replace_values(df_with_german_values=df, feature_name="notRepairedDamage", old_value="nein", new_value="no")

    return df


# Calculating new feature `age_of_car`
def get_age_of_car(df: pd.DataFrame) -> pd.DataFrame:
    df["lastSeen_year"] = pd.to_datetime(df["lastSeen"]).dt.year
    df["age_of_car"] = (df["lastSeen_year"] - df["yearOfRegistration"])

    return df


# Filtering data to consider only last 10 years of data
def get_last_10_years_old_cars(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["age_of_car"] <= 10]


# Converting some more columns from German to English values
def get_german_to_english_features(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = df["name"].str.replace('[^a-zA-Z0-9]', ' ')
    df["seller"] = df["seller"].replace(["privat"], "private")
    df["seller"] = df["seller"].replace(["gewerblich"], "commercial")

    df["vehicleType"] = df["vehicleType"].replace(["andere"], "others")

    df["fuelType"] = df["fuelType"].replace(["andere"], "others")
    df["fuelType"] = df["fuelType"].replace(["benzin"], "benzine")
    df["fuelType"] = df["fuelType"].replace(["elektro"], "electric")

    df["gearbox"] = df["gearbox"].replace(["manuell"], "manual")
    df["gearbox"] = df["gearbox"].replace(["automatik"], "automatic")

    return df


# Scale numeric features
def get_scaled_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[["powerPS_scaled"]] = scaler.fit_transform(df[["powerPS"]])
    df[["kilometer_scaled"]] = scaler.fit_transform(df[["kilometer"]])
    df[["age_of_car_scaled"]] = scaler.fit_transform(df[["age_of_car"]])

    return df


# One-hot encoding on categorical data
def get_encoded_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = [
        "seller",
        "offerType",
        "abtest",
        "vehicleType",
        "gearbox",
        "model",
        "fuelType",
        "brand",
        "notRepairedDamage"
    ]

    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return df


# main function to get cleaned data
def get_cleaned_data(data_path: str) -> pd.DataFrame:
    initial_selected_columns: List[str] = [
        "name",
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
        "dateCreated",
        "postalCode",
        "lastSeen"
    ]

    used_cars: pd.DataFrame = load_csv_data(input_path=data_path, columns=initial_selected_columns)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_data_without_null_values(df=used_cars)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_data_without_duplicate_records(df=used_cars)
    # logger.info(used_cars.info())

    columns_to_check_outliers = ["price", "kilometer", "powerPS"]
    used_cars: pd.DataFrame = get_data_without_outliers(df=used_cars, columns=columns_to_check_outliers)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = replace_feature_values(df=used_cars)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_age_of_car(df=used_cars)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_last_10_years_old_cars(df=used_cars)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_german_to_english_features(df=used_cars)
    # logger.info(used_cars.info())

    used_cars: pd.DataFrame = get_scaled_data(df=used_cars)
    # logger.info(used_cars.info())

    features_to_select = [
        "seller",
        "offerType",
        "abtest",
        "vehicleType",
        "gearbox",
        "model",
        "fuelType",
        "brand",
        "notRepairedDamage",
        "age_of_car_scaled",
        "powerPS_scaled",
        "kilometer_scaled",
        "price"
    ]
    used_cars = used_cars[features_to_select]

    encoded_used_cars = get_encoded_data(df=used_cars)
    # logger.info(encoded_used_cars.shape)

    return encoded_used_cars
