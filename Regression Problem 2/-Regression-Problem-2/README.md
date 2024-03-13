# -Regression-Problem-2


## Project Overview
This project is a machine learning application designed to predict the lifespan of organisms based on data from three sensors. It utilizes Support Vector Regression (SVR) and Linear Regression models to make predictions. The app is built with Streamlit, allowing users to interactively upload training and testing datasets, train models, and view prediction results.

## Features
- **Data Upload:** Users can upload their own CSV files for training and testing the models.
- **Model Training and Evaluation:** Automatically trains SVR and Linear Regression models and evaluates them using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Interactive UI:** Easy-to-use interface for non-technical users to utilize machine learning predictions.

## Installation
To set up and run this application locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/lifespan-prediction-app.git
   cd lifespan-prediction-app

2.**Install Requirements**
Ensure you have Python installed on your system. Then install the required Python packages using:   
    
    pip install -r requirements.txt

3.**Run the Application**
Launch the app by running:
    
    streamlit run lifespan_prediction_app.py

## Usage
After starting the app, follow these steps to use it:

Navigate to the URL provided by Streamlit in your browser.
Use the file uploader to select and upload your training and testing datasets in CSV format.
The app will automatically process the uploaded files, train the models, and display the prediction results.
Review the MSE and MAE metrics for both SVR and Linear Regression models.

## Contributing
We welcome contributions to improve this project. If you have suggestions or bug reports, please open an issue in the repository. For major changes, please open a pull request for review.

