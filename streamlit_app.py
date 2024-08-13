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
if selected_tab[0]:
    st.header("Data Ingestion")
    st.write("Enter the complete path where the source data is stored.")
    path = st.text_input("Path of the file")
    def upload_file():
      """
      This function handles the upload of a file using a button and drag and drop.
      """
      uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], accept_multiple_files=False)
      if uploaded_file is not None:
        # Handle the uploaded file
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
          f.write(uploaded_file.read())
        st.success("File uploaded successfully!")
      else:
        st.info("Please upload a file.")
    
    # Call the function to create the file upload button and handle the file
    upload_file()

    st.write("Enter the complete name with extension of the source data i.e..csv or .xlsx")
    name = st.text_input("Name of the file")

    st.write("Data dimensions")
    data_dimensions = st.button("Run")

    if data_dimensions:
        st.write("Confirmation message")

# Data Transformation tab
if selected_tab[1]:
    st.header("Data Transformation")
    st.write("This is the Data Transformation tab.")

