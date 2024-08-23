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



        # Create a pie chart
        st.header("Pie Chart of the Data")
        
        # Get the column names
        columns = data.columns.tolist()
        
        # Create a dropdown to select the column for the pie chart
        pie_column = st.selectbox("Select the column for the pie chart", columns)
        
        # Create a pie chart
        pie_chart_data = data[pie_column].value_counts().to_dict()
        
        st.write(pie_chart_data)
        
        nivo.pie(pie_chart_data, height=400, width=800)

        
    
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



            # Display the predictions
            st.write("Predictions:")
            st.write(y_pred)
