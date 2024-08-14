import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(
    page_title="ML model",
    page_icon=":rocket:",
    layout="wide",
)

# Data Ingestion tab
st.header("Data Ingestion")

# Create a file uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded_file:
    # Create the uploads directory if it doesn't exist
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Handle the uploaded file
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")

    # Get the file name and path
    file_name = uploaded_file.name
    file_path = file_path

    # Display the file name and path
    st.write(f"File name: {file_name}")
    st.write(f"File path: {file_path}")

    # Load the uploaded data
    if file_name.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_name.endswith('.xlsx'):
        data = pd.read_excel(file_path)

    # Display the data dimensions
    st.write(f"Data shape: {data.shape}")

    # Load the pre-trained model
    with open(r'linear_reg_model (1).pkl', 'rb') as handle:
        model = pickle.load(handle)

    # Get the column names
    columns = data.columns.tolist()

    # Create a dropdown to select the target column
    target_column = st.selectbox("Select the target column", columns)

    # Split the data into features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

  # Make predictions on the uploaded data
    X_array = X.values  # Convert X to a NumPy array
    n_features = X_array.shape[1]
    expected_n_features = 5  # Replace with the number of features your model expects
    if n_features != expected_n_features:
        raise ValueError(f"Model expects {expected_n_features} features, but got {n_features}")
    y_pred = model.predict(X_array)

    # Calculate the accuracy score (R-squared)
    r2 = r2_score(y, y_pred)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)

    # Display the accuracy score (R-squared)
    st.write(f"R-squared score: {r2:.3f}")

    # Display the Mean Squared Error (MSE)
    st.write(f"Mean Squared Error (MSE): {mse:.3f}")

    # Display the loaded model
    st.write(model)
