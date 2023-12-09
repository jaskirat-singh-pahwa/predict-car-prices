import streamlit as st
import psycopg2
import pandas as pd


db_params = {
    'host': 'localhost',
    'database': 'Airbnb_listings',
    'user': 'postgres',
    'password': 'Pass@12345',
}


def fetch_data(query):
    connection = psycopg2.connect(**db_params)
    df = pd.read_sql(query, connection)
    connection.close()
    return df


def main() -> None:
    st.title("PostgreSQL Data Display")

    # Example: Fetch and display data from a table
    table_name = "neighbourhood"
    query = f"SELECT * FROM {table_name};"

    data = fetch_data(query)

    st.write(f"Displaying data from {table_name} table:")
    st.write(data)


if __name__ == "__main__":
    main()
