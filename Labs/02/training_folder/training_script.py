from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os
import argparse

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

try:
    # Prepare the dataset
    diabetes = pd.read_csv('diabetes.csv')
    
    # Separate features and labels
    X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
                     'SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Train a logistic regression model
    run.log('Regularization', reg)  # Fixed: log name-value pair properly
    print(f'Training a logistic regression model with regularization rate of {reg}')
    
    model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

    # Calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print(f'Accuracy: {acc}')
    
    # Calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print(f'AUC: {auc}')
    
    # Log metrics properly
    run.log('Accuracy', float(acc))  # Fixed: use float() instead of np.float()
    run.log('AUC', auc)  # Fixed: log as name-value pair

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/model.pkl'
    joblib.dump(value=model, filename=model_path)
    print(f'Model saved to {model_path}')
    
    # Register the model (optional but recommended)
    run.upload_file(name='model.pkl', path_or_stream=model_path)

except Exception as e:
    run.log('Error', str(e))
    print(f'Error during training: {e}')
    run.fail(error_details=str(e))
    raise

finally:
    run.complete()