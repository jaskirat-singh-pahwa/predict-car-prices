import streamlit as st

import pandas as pd
from runner import run_app


def run_streamlit_app():
    st.title("Machine Learning Price Prediction")

    # file_path = r"C:\Users\Sree Lekshmi P\DIC_project\predict-car-prices-master\input\raw\autos.csv"

    file_path = st.text_input("Enter the path of your CSV file:")

    if file_path:
        print(type(file_path), file_path)
        df = pd.read_csv(file_path)
        print(df.head())

        if st.button("Run Prediction"):
            if file_path:
                try:
                    output = run_app(user_input_file_path=file_path)
                    # predictions_path = "output/user_data_predictions.csv"  # Replace with the actual path
                    # predictions_df = pd.read_csv(predictions_path)

                    st.write("Predictions:")
                    st.write(output)
                except Exception as e:
                    st.write(f"Error running predictions: {e}")
            else:
                st.write("Please enter a valid file path.")
    
        if st.button("Show Data"):
            if file_path:
                try:
                    data_df = pd.read_csv(file_path)
                    st.write("Data from the provided file:")
                    st.write(data_df)
                except Exception as e:
                    st.write(f"Error reading the file: {e}")
            else:
                st.write("Please enter a file path to show the data.")
