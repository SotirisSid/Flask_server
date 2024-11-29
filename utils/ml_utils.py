import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from models import db, PreprocessedKeystrokeData, User  # Import your models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier  # Importing MLPClassifier
import joblib  # Importing joblib for saving models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.utils import check_array

def train_model(preprocessed=True):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    
    try:
        # Load preprocessed data
        data = pd.read_sql('SELECT * FROM preprocessed_keystroke_data', session.bind)
        
        # Define expected columns for features
        expected_columns = [
            'press_press_interval_mean',
            'release_interval_mean',
            'hold_time_mean',
            'press_press_interval_variance',
            'release_interval_variance',
            'hold_time_variance',
            'backspace_count',
            'error_rate',
            'total_typing_time',
            'typing_speed_cps'
        ]
        
        # Define weights for each feature
        feature_weights = {
            'press_press_interval_mean': 1.5,
            'release_interval_mean': 1.2,
            'hold_time_mean': 1.2,
            'press_press_interval_variance': 1.5,
            'release_interval_variance': 1.3,
            'hold_time_variance': 1.3,
            'backspace_count': 1.0,
            'error_rate': 1.0,
            'total_typing_time': 1.1,
            'typing_speed_cps': 1.4,
        }
        
        # Extract features and target variable
        X = data[expected_columns].copy()
        y = data['user_id']
        
        # Apply weights to the features
        for feature in expected_columns:
            X[feature] *= feature_weights[feature]

        # Standardize features and keep them as a DataFrame
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Save scaler for consistent scaling during predictions
        scaler_path = os.path.join('models', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)

        # Define and train models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000), # No depth (linear model).
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'), #Depth of individual trees controlled by max_depth.
            'Support Vector Machine': SVC(probability=True), #No depth in the traditional sense; complexity depends on the kernel.
            'Gradient Boosting': GradientBoostingClassifier(), #Depth of individual trees controlled by max_depth
            'Neural Network': MLPClassifier(max_iter=1000) #Depth controlled by the number of hidden layers in hidden_layer_sizes.
        }
        
        results = {}
        for model_name, model in models.items():
            print(f"Training model: {model_name}")  # Debugging line
            model.fit(X_scaled, y)  # Ensure X_scaled is a DataFrame
            model_path = os.path.join('models', f'{model_name.replace(" ", "_").lower()}.joblib')
            joblib.dump(model, model_path)
            results[model_name] = "Model trained and saved successfully"
        
        return results

    finally:
        session.close()
def train_model_with_username(preprocessed=True):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    try:
        data = pd.read_sql('SELECT * FROM preprocessed_keystroke_data', session.bind)
        # Define expected columns including username features
        expected_columns = [
            'press_press_interval_mean',
            'release_interval_mean', 
            'hold_time_mean',
            'press_press_interval_variance',
            'release_interval_variance',
            'hold_time_variance',
            'backspace_count',
            'error_rate', 
            'total_typing_time',
            'typing_speed_cps',
            # Username features
            'username_press_press_interval_mean',
            'username_release_interval_mean',
            'username_hold_time_mean',
            'username_press_press_interval_variance', 
            'username_release_interval_variance',
            'username_hold_time_variance',
            'username_backspace_count',
            'username_error_rate',
            'username_total_typing_time',
            'username_typing_speed_cps'
        ]
        
        # Define weights for each feature
        feature_weights = {
            # Password features
            'press_press_interval_mean': 1.5,
            'release_interval_mean': 1.2,
            'hold_time_mean': 1.2,
            'press_press_interval_variance': 1.5,
            'release_interval_variance': 1.3,
            'hold_time_variance': 1.3,
            'backspace_count': 1.0,
            'error_rate': 1.0,
            'total_typing_time': 1.1,
            'typing_speed_cps': 1.4,
            # Username features (slightly lower weights)
            'username_press_press_interval_mean': 1.3,
            'username_release_interval_mean': 1.0,
            'username_hold_time_mean': 1.0,
            'username_press_press_interval_variance': 1.3,
            'username_release_interval_variance': 1.1,
            'username_hold_time_variance': 1.1,
            'username_backspace_count': 0.8,
            'username_error_rate': 0.8,
            'username_total_typing_time': 0.9,
            'username_typing_speed_cps': 1.2
        }

        # Extract features and target variable
        X = data[expected_columns].copy()
        y = data['user_id']
        
        # Apply weights to the features
        for feature in expected_columns:
            X[feature] *= feature_weights[feature]

        # Standardize features and keep them as a DataFrame
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Save scaler for consistent scaling during predictions
        scaler_path = os.path.join('models', 'scaler_username.joblib')
        joblib.dump(scaler, scaler_path)

        # Define and train models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000), # No depth (linear model).
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'), #Depth of individual trees controlled by max_depth.
            'Support Vector Machine': SVC(probability=True), #No depth in the traditional sense; complexity depends on the kernel.
            'Gradient Boosting': GradientBoostingClassifier(), #Depth of individual trees controlled by max_depth
            'Neural Network': MLPClassifier(max_iter=1000) #Depth controlled by the number of hidden layers in hidden_layer_sizes.
        }
        
        results = {}
        for model_name, model in models.items():
            print(f"Training model: {model_name}")  # Debugging line
            model.fit(X_scaled, y)  # Ensure X_scaled is a DataFrame
            model_path = os.path.join('models', f'{model_name.replace(" ", "_").lower()+"_with_username"}.joblib')
            joblib.dump(model, model_path)
            results[model_name] = "Model trained and saved successfully"
        
        return results
    finally:
        session.close()