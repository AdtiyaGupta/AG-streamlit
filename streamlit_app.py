import streamlit as st
import streamlit_antd_components as sac
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="ML model",
    page_icon=":rocket:",
    layout="wide",
)

#Menu Bar
with st.sidebar:
    selected = sac.menu([
        sac.MenuItem('home', icon='house-fill'),
        sac.MenuItem(type='divider'),
        sac.MenuItem('products', icon='box-fill', children=[
            sac.MenuItem('Data Ingestion'),           
            sac.MenuItem('Data Transformation', icon='', description=''),
            sac.MenuItem('Auto Train ML Model', icon=''),
            sac.MenuItem('Freeze the Learning', icon=''),
        ]),
        sac.MenuItem('disabled', disabled=True),
        sac.MenuItem(type='divider'),
        sac.MenuItem('link', type='group', children=[
            sac.MenuItem('@1', icon='', href=''),
            sac.MenuItem('@2', icon='', href=''),
        ]),
    ], size='xl', variant='left-bar', color='grape', open_all=True, return_index=True)


#Home bar
if selected == 0:
    st.header("Welcome to ML Model")

    st.write("This is a machine learning model that allows you to upload your dataset, select the target column, and train a simple linear regression model. The model will then make predictions on the uploaded data.")

    st.subheader("Features")

    st.write("The following features are available in this model:")

    features = [
        "Data Ingestion: Upload your dataset in CSV or Excel format",
        "Target Column Selection: Select the column you want to predict",
        "Model Training: Train a simple linear regression model on your data",
        "Predictions: Get predictions on your uploaded data"
    ]

    for feature in features:
        st.write(f"* {feature}")

    st.subheader("How it Works")

    st.write("Here's a step-by-step guide on how to use this model:")

    steps = [
        "Upload your dataset using the file uploader",
        "Select the target column from the dropdown menu",
        "Click the 'Train Model' button to train the model",
        "Get predictions on your uploaded data"
    ]

    for step in steps:
        st.write(f"* {step}")

    st.subheader("Benefits")

    st.write("Using this model, you can:")

    benefits = [
        "Quickly upload and analyze your dataset",
        "Select the target column with ease",
        "Train a simple linear regression model with minimal effort",
        "Get accurate predictions on your uploaded data"
    ]

    for benefit in benefits:
        st.write(f"* {benefit}")
    
    
# Data Ingestion tab
if selected == 3:
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
    
        # Define the model file path
        model_file_path = "linear_reg_model(1).pkl"
    
        # Get the column names
        columns = data.columns.tolist()
    
        # Create a dropdown to select the target column
        target_column = st.selectbox("Select the target column", columns)
    
        # Select the correct features
        features = [col for col in columns if col != target_column]
    
        # Define X as the feature columns
        X = data[features]
    
        # Define y as the target column
        y = data[target_column]
    
        # Check if y is numeric
        if y.dtype.kind not in 'bifc':
            # Convert y to numeric using LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
    
        # Train and save the model if it doesn't exist
        if not os.path.exists(model_file_path):
            # Handle missing values
            numeric_features = [col for col in X.columns if X[col].dtype.kind in 'bifc']
    
            if not numeric_features:
                st.error("No numeric features found in the data. Please check your data and try again.")
            else:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                ])
    
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                    ]
                )
    
                # Train a simple linear regression model
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())])
                model.fit(X, y)
    
                # Save the model to a file
                with open(model_file_path, 'wb') as handle:
                    pickle.dump(model, handle)
    
        # Load the pre-trained model
        with open(model_file_path, 'rb') as handle:
            model = pickle.load(handle)
    
        # Get the column names expected by the ColumnTransformer
        expected_columns = model.named_steps['preprocessor'].transformers_[0][2]
    
        # Check if all expected columns are present in X
        if not all(col in X.columns for col in expected_columns):
            missing_columns = [col for col in expected_columns if col not in X.columns]
            st.error(f"Columns are missing: {missing_columns}. Please check your data and try again.")
        else:
            # Make predictions on the uploaded data
            y_pred = model.predict(X)
    
            # Calculate the accuracy score (R-squared)
            r2 = r2_score(y, y_pred)
            
            # Calculate the Mean Squared Error (MSE)
            mse = mean_squared_error(y, y_pred)
    
            sac.divider(label='Result', icon='result', align='center', color='gray')
            
            # Display the accuracy score (R-squared)
            st.write(f"R-squared: {r2:.2f}")
    
            
            
            # Display the Mean Squared Error (MSE)
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")


import json

from streamlit_elements import nivo, mui
from .dashboard import Dashboard


class Pie(Dashboard.Item):

    DEFAULT_DATA = [
        { "id": "java", "label": "java", "value": 465, "color": "hsl(128, 70%, 50%)" },
        { "id": "rust", "label": "rust", "value": 140, "color": "hsl(178, 70%, 50%)" },
        { "id": "scala", "label": "scala", "value": 40, "color": "hsl(322, 70%, 50%)" },
        { "id": "ruby", "label": "ruby", "value": 439, "color": "hsl(117, 70%, 50%)" },
        { "id": "elixir", "label": "elixir", "value": 366, "color": "hsl(286, 70%, 50%)" }
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theme = {
            "dark": {
                "background": "#252526",
                "textColor": "#FAFAFA",
                "tooltip": {
                    "container": {
                        "background": "#3F3F3F",
                        "color": "FAFAFA",
                    }
                }
            },
            "light": {
                "background": "#FFFFFF",
                "textColor": "#31333F",
                "tooltip": {
                    "container": {
                        "background": "#FFFFFF",
                        "color": "#31333F",
                    }
                }
            }
        }

    def __call__(self, json_data):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            data = self.DEFAULT_DATA

        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar():
                mui.icon.PieChart()
                mui.Typography("Pie chart", sx={"flex": 1})

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                nivo.Pie(
                    data=data,
                    theme=self._theme["dark" if self._dark_mode else "light"],
                    margin={ "top": 40, "right": 80, "bottom": 80, "left": 80 },
                    innerRadius=0.5,
                    padAngle=0.7,
                    cornerRadius=3,
                    activeOuterRadiusOffset=8,
                    borderWidth=1,
                    borderColor={
                        "from": "color",
                        "modifiers": [
                            [
                                "darker",
                                0.2,
                            ]
                        ]
                    },
                    arcLinkLabelsSkipAngle=10,
                    arcLinkLabelsTextColor="grey",
                    arcLinkLabelsThickness=2,
                    arcLinkLabelsColor={ "from": "color" },
                    arcLabelsSkipAngle=10,
                    arcLabelsTextColor={
                        "from": "color",
                        "modifiers": [
                            [
                                "darker",
                                2
                            ]
                        ]
                    },
                    defs=[
                        {
                            "id": "dots",
                            "type": "patternDots",
                            "background": "inherit",
                            "color": "rgba(255, 255, 255, 0.3)",
                            "size": 4,
                            "padding": 1,
                            "stagger": True
                        },
                        {
                            "id": "lines",
                            "type": "patternLines",
                            "background": "inherit",
                            "color": "rgba(255, 255, 255, 0.3)",
                            "rotation": -45,
                            "lineWidth": 6,
                            "spacing": 10
                        }
                    ],
                    fill=[
                        { "match": { "id": "ruby" }, "id": "dots" },
                        { "match": { "id": "c" }, "id": "dots" },
                        { "match": { "id": "go" }, "id": "dots" },
                        { "match": { "id": "python" }, "id": "dots" },
                        { "match": { "id": "scala" }, "id": "lines" },
                        { "match": { "id": "lisp" }, "id": "lines" },
                        { "match": { "id": "elixir" }, "id": "lines" },
                        { "match": { "id": "javascript" }, "id": "lines" }
                    ],
                    legends=[
                        {
                            "anchor": "bottom",
                            "direction": "row",
                            "justify": False,
                            "translateX": 0,
                            "translateY": 56,
                            "itemsSpacing": 0,
                            "itemWidth": 100,
                            "itemHeight": 18,
                            "itemTextColor": "#999",
                            "itemDirection": "left-to-right",
                            "itemOpacity": 1,
                            "symbolSize": 18,
                            "symbolShape": "circle",
                            "effects": [
                                {
                                    "on": "hover",
                                    "style": {
                                        "itemTextColor": "#000"
                                    }
                                }
                            ]
                        }
                    ]
                )


            
            
            # Display the predictions
            st.write("Predictions:")
            st.write(y_pred)
