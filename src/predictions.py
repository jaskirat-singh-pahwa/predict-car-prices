import joblib
import pandas as pd


"""
This module is used for running predictions on user given data.
As we can notice in the get_loaded_model function, we will be using gbr to make our predictions.
"""


def get_loaded_model():
    with open(f"models/fine_tuned/gbr.pkl", "rb") as file:
        loaded_model = joblib.load(file)
        return loaded_model


def get_predicted_output(
        raw_user_data: pd.DataFrame,
        processed_user_data: pd.DataFrame
) -> pd.DataFrame:
    trained_model = get_loaded_model()
    df_without_target = processed_user_data.loc[:, processed_user_data.columns != "price"]
    df_without_target = df_without_target.drop("Unnamed: 0", axis=1, errors='ignore')

    predictions = trained_model.predict(df_without_target)
    raw_user_data["predictions"] = predictions

    # Saving predicted output in the output directory
    raw_user_data.to_csv(path_or_buf="output/user_data/user_data_predictions.csv", index=False)

    return raw_user_data
