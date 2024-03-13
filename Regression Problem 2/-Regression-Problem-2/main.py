import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title('Lifespan Prediction')

# File uploader widgets for training and testing datasets
train_file = st.file_uploader("Choose a training dataset file (CSV format)", type=['csv'])
test_file = st.file_uploader("Choose a testing dataset file (CSV format)", type=['csv'])

# Function to load and preprocess data
def load_and_preprocess_data(train_file, test_file):
    # Read CSV files
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Assign consistent column names
    new_column_names = ['Sensor1', 'Sensor2', 'Sensor3', 'Target']
    train_df.columns = new_column_names
    test_df.columns = new_column_names
    
    return train_df, test_df

# Main app logic
if train_file is not None and test_file is not None:
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data(train_file, test_file)
    
    # Separate features and target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train models
    models = {'SVR': SVR(kernel='rbf'), 'Linear Regression': LinearRegression()}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        results[name] = {'MSE': mse, 'MAE': mae}
    
    # Display results
    for model_name, metrics in results.items():
        st.write(f"**{model_name}**")
        st.write(f"Mean Squared Error (MSE): {metrics['MSE']}")
        st.write(f"Mean Absolute Error (MAE): {metrics['MAE']}")

else:
    st.write("Please upload both the training and testing dataset files to begin.")
