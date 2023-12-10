import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from runner import run_app
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import seaborn as sns


def run_streamlit_app():
    # Page layout and custom styles
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        .reportview-container {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)
    sns.set_theme(style="whitegrid")  # theme for seaborn plots

    st.markdown('<p class="big-font">Used Car Price Prediction Using Gradient Boosting Regressor</p>',
                unsafe_allow_html=True)

    # Input for CSV file path
    file_path = st.text_input("Enter the path of your CSV file:")
    # Process the input file and display predictions
    if file_path:
        df = pd.read_csv(file_path)
        print(df.head())

        if st.button("Run Prediction"):
            if file_path:
                try:
                    # Run the prediction model
                    output = run_app(user_input_file_path=file_path)
                    st.write("Predictions:")
                    st.dataframe(output, height=300)

                    # Display top brand and top 3 models
                    st.markdown("""
                        <style>
                        .top-brand-models {
                            font-size: 20px;
                            font-weight: bold;
                            margin-top: 10px;
                            margin-bottom: 10px;
                            color: #4CAF50;
                            border-left: 5px solid #4CAF50;
                            padding-left: 10px;
                        }
                        .model-list {
                            font-size: 18px;
                            margin-left: 20px;
                            list-style-type: circle;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                    st.markdown("<div class='top-brand-models'>Top Brand and Models 🚗</div>", unsafe_allow_html=True)

                    top_brand = output.groupby('brand').size().sort_values(ascending=False).head(1)
                    st.markdown(f"<div class='top-brand-models'>Top Brand: <b>{top_brand.index[0]}</b></div>",
                                unsafe_allow_html=True)

                    top_models = output.groupby('model').size().sort_values(ascending=False).head(3)
                    st.markdown("<div class='top-brand-models'>Top 3 Models:</div>", unsafe_allow_html=True)
                    st.markdown("<ul class='model-list'>", unsafe_allow_html=True)
                    for model in top_models.index:
                        st.markdown(f"<li>{model}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)

                    # Plotting average actual price by brand
                    with st.expander("Trend with Brand"):
                        col1, col2 = st.columns(2)
                        # Group by 'brand' and calculate mean for 'price' and 'predictions'
                        grouped_data = output.groupby('brand').agg({'price': 'mean', 'predictions': 'mean'})
                        # Reset index to make 'age_of_car' a column again for plotting)
                        grouped_data = grouped_data.reset_index()

                        with col1:
                            # Trend with Brand with Price
                            fig, ax = plt.subplots()
                            sns.barplot(x='brand', y='price', data=grouped_data, palette='coolwarm', ax=ax)
                            ax.set_title('Average Actual Price by Brand', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Brand', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Average Price', fontsize=12, fontweight='bold')
                            ax.tick_params(axis='x', rotation=45)
                            st.pyplot(fig)

                        with col2:
                            # Trend with Brand with Predicted Price
                            fig, ax = plt.subplots()
                            sns.barplot(x='brand', y='predictions', data=grouped_data, palette='coolwarm', ax=ax)
                            ax.set_title('Average Predicted Price by Brand', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Brand', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Average Predicted Price', fontsize=12, fontweight='bold')
                            ax.tick_params(axis='x', rotation=45)
                            st.pyplot(fig)

                    # Plotting average actual price by Age of Car
                    with st.expander("Trend with Age of Car"):
                        col1, col2 = st.columns(2)

                        # Group by 'age_of_car' and calculate mean for 'price' and 'predictions'
                        grouped_by_age = output.groupby('age_of_car').agg({'price': 'mean', 'predictions': 'mean'})

                        # Reset index to make 'age_of_car' a column again (useful for plotting)
                        grouped_by_age = grouped_by_age.reset_index()
                        with col1:
                            # Trend with Age of Car with Price
                            fig, ax = plt.subplots()
                            sns.barplot(x='age_of_car', y='price', data=grouped_by_age, ax=ax, label='Actual Price',
                                        color='skyblue')
                            ax.set_title('Average Actual Price by Age of Car', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Age of Car (Years)', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Average Price', fontsize=12, fontweight='bold')
                            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                            st.pyplot(fig)

                        with col2:
                            # Trend with Age of Car with Price
                            fig, ax = plt.subplots()
                            sns.barplot(x='age_of_car', y='predictions', data=grouped_by_age, ax=ax,
                                        label='Predicted Price', color='lightgreen')
                            ax.set_title('Average Predicted Price by Age of Car', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Age of Car (Years)', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Average Predicted Price', fontsize=12, fontweight='bold')
                            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                            st.pyplot(fig)

                    # Actual Vs Predicted
                    with st.expander("View Actual vs Predicted Comparison"):
                        fig, ax = plt.subplots()
                        ax.scatter(output['price'], output['predictions'], alpha=0.7, edgecolors='w', s=100)
                        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
                        ax.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
                        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                        st.pyplot(fig)

                    # Model Performance Metrics
                    mae = mean_absolute_error(output['price'], output['predictions'])
                    mse = mean_squared_error(output['price'], output['predictions'])
                    rmse = np.sqrt(mse)
                    r2 = r2_score(output['price'], output['predictions'])

                    # Creating a DataFrame for the metrics
                    metrics_data = {
                        'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
                        'Value': [mae, mse, rmse, r2]
                    }
                    metrics_df = pd.DataFrame(metrics_data)

                    # Displaying the DataFrame
                    st.write("Model Performance Metrics:")
                    st.dataframe(metrics_df.style.format({"Value": "{:.2f}"}))

                except Exception as e:
                    st.error(f"Error running predictions: {e}")
            else:
                st.warning("Please enter a valid file path.")

        # Show Data Section for input csv
        if st.button("Show Data"):
            if file_path:
                try:
                    data_df = pd.read_csv(file_path)
                    with st.expander("Data from the provided file:"):
                        st.write(data_df)
                except Exception as e:
                    st.error(f"Error reading the file: {e}")
            else:
                st.warning("Please enter a file path to show the data.")
    st.markdown('<p class="footer">Sree Lekshmi<br> Gayathri <br>Jaskirat </p>', unsafe_allow_html=True)
