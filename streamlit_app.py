import streamlit as st
#from PIL import Image

# Set page config
st.set_page_config(
    page_title="Preprod Emulet",
    page_icon=":rocket:",
    layout="wide",
)

# Load image
#image = Image.open("preprod_emulet.png")
#st.image(image, use_column_width=True)

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

# Auto Train ML models tab
if selected_tab[2]:
    st.header("Auto Train ML models")
    st.write("This is the Auto Train ML models tab.")

# Freeze the learnings tab
if selected_tab[3]:
    st.header("Freeze the learnings")
    st.write("This is the Freeze the learnings tab.")
