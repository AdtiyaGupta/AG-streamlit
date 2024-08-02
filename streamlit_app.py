import streamlit as st
import os


# Set page config
st.set_page_config(
    page_title="Preprod Emulet",
    page_icon=":rocket:",
    layout="wide",
)



# Add sidebar
st.sidebar.header("Preprod Emulet")
st.sidebar.write("This is a demo of a Streamlit app.")

# Add tabs
tabs = ["Data Ingestion", "Data Transformation", "Auto Train ML models", "Freeze the learnings"]
selected_tab = st.tabs(tabs)

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
    class DataTransformation:
    def __init__(self, data):
        self.data = data

    def aggregate(self, aggregation_type):
        # Apply aggregation transformation
        if aggregation_type == 'sum':
            return self.data.sum()
        elif aggregation_type == 'mean':
            return self.data.mean()
        else:
            raise ValueError('Invalid aggregation type')

    def filter(self, filter_condition):
        # Apply filter transformation
        return self.data[filter_condition]

    def sort(self, sort_column):
        # Apply sort transformation
        return self.data.sort_values(by=sort_column)

# Example usage:
data = pd.read_csv('data.csv')
transformation = DataTransformation(data)
transformed_data = transformation.aggregate('sum')
print(transformed_data)

# Auto Train ML models tab
if selected_tab[2]:
    st.header("Auto Train ML models")
    st.write("This is the Auto Train ML models tab.")

# Freeze the learnings tab
if selected_tab[3]:
    st.header("Freeze the learnings")
    st.write("This is the Freeze the learnings tab.")
