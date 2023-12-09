import joblib
import pandas as pd


def get_loaded_model():
    with open("models/fine_tuned/gbr.pkl", "rb") as file:
        loaded_model = joblib.load(file)
        return loaded_model


def get_predicted_output(df: pd.DataFrame) -> pd.DataFrame:
    trained_model = get_loaded_model()
    df_without_target = df.loc[:, df.columns != "price"]
    df_without_target = df_without_target.drop("Unnamed: 0", axis=1, errors='ignore')

    predictions = trained_model.predict(df_without_target)
    df["predictions"] = predictions

    df.to_csv(path_or_buf="output/user_data/user_data_predictions.csv", index=False)

    return df
