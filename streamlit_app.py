import streamlit as st
import os
import pickle

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
    # Handle the uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")

    # Get the file name and path
    file_name = uploaded_file.name
    file_path = file_path

    # Display the file name and path
    st.write(f"File name: {file_name}")
    st.write(f"File path: {file_path}")

    # Display the data dimensions
    if st.button("Run"):
        st.write("Confirmation message")

# Data Transformation tab
st.header("Data Transformation")
st.write("This is the Data Transformation tab.")

# Load the pre-trained model
with open('linear_reg_model (1).pkl', 'rb') as handle:
    model = pickle.load(handle)

# Display the loaded model
st.write(model)
