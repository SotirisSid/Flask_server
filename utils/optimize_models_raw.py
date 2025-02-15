import joblib
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from skopt.space import Real, Categorical
from sqlalchemy.sql import text
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV  # Bayesian optimization for hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import ast  # To safely evaluate the string representation of lists

# Path to the database
db_path = r'D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db'
scaler_path = r'D:\THESIS\MobileApp\Flask_server\models\raw_scaler.joblib'  # Use the existing scaler

# Create an SQLAlchemy engine to connect to the SQLite database
engine = create_engine(f'sqlite:///{db_path}', echo=True)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Correct expected columns as per your request
expected_columns = [
    'press_press_intervals',
    'release_press_intervals',
    'hold_times',
    'total_typing_time',
    'typing_speed',
    'press_to_release_ratio_mean'  # New feature
]

import pandas as pd
import numpy as np
import joblib
import ast
from sqlalchemy.sql import text

# Load the scaler from the specified path
scaler = joblib.load(scaler_path)

# Define the expected columns
expected_columns = [
    'press_press_intervals', 'release_press_intervals', 'hold_times', 
    'total_typing_time', 'typing_speed', 'press_to_release_ratio_mean'
]

# Define the fixed max password length and max feature length for padding
max_password_len = 18  # Maximum length of a password (e.g., 8-9 characters)
max_feature_length = 150  # Adjust as per your data requirements



def load_data(session, table_name, max_entries=None, use_all_data=True):
    # Load data from the database
    data = pd.read_sql(f'SELECT * FROM {table_name}', session.bind)

    # Exclude admin user (user_id == 1)
    data = data[data['user_id'] != 1]

    # Group data by user_id
    user_groups = data.groupby('user_id')

    # Check if each user has enough data
    if max_entries is not None:
        for user_id, group in user_groups:
            if len(group) < max_entries:
                return {"error": f"User {user_id} has only {len(group)} entries, which is less than the required {max_entries} entries."}

    # Limit data to max_entries per user if required
    if not use_all_data and max_entries is not None:
        data = data.groupby('user_id').head(max_entries)

    features = []
    labels = []

    # Loop through the data and construct feature vectors
    for _, row in data.iterrows():
        try:
            # Parse stringified lists into actual Python lists
            press_press_intervals = eval(row['press_press_intervals'])
            release_press_intervals = eval(row['release_press_intervals'])
            hold_times = eval(row['hold_times'])

            # Check if the sequence is excessively long (junk data)
            total_length = len(press_press_intervals) + len(release_press_intervals) + len(hold_times)
            if total_length > max_feature_length:  # This threshold indicates junk data
                print(f"Skipping user {row['user_id']} due to excessive keystroke length.")
                continue  # Skip this row if data is too long (junk)

            # Pad feature vectors to the fixed max password length
            press_press_intervals = press_press_intervals[:max_password_len] + [0] * (max_password_len - len(press_press_intervals))
            release_press_intervals = release_press_intervals[:max_password_len] + [0] * (max_password_len - len(release_press_intervals))
            hold_times = hold_times[:max_password_len] + [0] * (max_password_len - len(hold_times))

            # Construct the feature vector by combining them in a fixed order
            feature_vector = (
                press_press_intervals +
                release_press_intervals +
                hold_times +
                [
                    row['total_typing_time'],
                    row['typing_speed'],
                    row['error_rate'],
                    row['press_to_release_ratio_mean']
                ]
            )

            # Convert feature vector values to float and handle NaN/inf
            feature_vector = [float(x) if pd.notna(x) and np.isfinite(x) else 0.0 for x in feature_vector]
            features.append(feature_vector)
            labels.append(row['user_id'])  # Add the user_id as the target variable (y)
        except Exception as e:
            print(f"Error processing row {row['id']}: {str(e)}")
            continue

    # Convert features to numpy array
    X = pd.DataFrame(features)
    y = pd.Series(labels)

    # Check shape before scaling
    print(f"Shape of X before scaling: {X.shape}")

    # Scale the features using the pre-trained scaler
    X_scaled = scaler.transform(X)

    # Now create a DataFrame with the scaled values, and generate column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Convert labels (y) to numpy array
    y = np.array(labels)

    # Return the scaled features and labels (y)
    return X_scaled_df, y



# Model Tuning Functions (same as before)
def tune_logistic_regression(X, y):
    param_space = {
        'C': (0.01, 100.0, 'uniform'),
        'solver': ['saga'],
        'penalty': ['l1', 'l2'],
    }
    model = LogisticRegression(max_iter=10000)
    bayes_search = BayesSearchCV(model, param_space, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X, y)
    return bayes_search.best_params_, bayes_search.best_score_

def tune_random_forest(X, y):
    param_space = {
        'n_estimators': (50, 200, 'uniform'),
        'max_depth': (1, 30),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
    }
    model = RandomForestClassifier(class_weight='balanced')
    bayes_search = BayesSearchCV(model, param_space, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X, y)
    return bayes_search.best_params_, bayes_search.best_score_

def tune_svc(X, y):
    param_space = {
        'C': (0.1, 100, 'uniform'),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
    }
    model = SVC(probability=True)
    bayes_search = BayesSearchCV(model, param_space, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X, y)
    return bayes_search.best_params_, bayes_search.best_score_

def tune_gradient_boosting(X, y):
    param_space = {
        'n_estimators': (50, 200, 'uniform'),
        'learning_rate': (0.01, 0.2, 'uniform'),
        'max_depth': (3, 7),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
    }
    model = GradientBoostingClassifier()
    bayes_search = BayesSearchCV(model, param_space, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    bayes_search.fit(X, y)
    return bayes_search.best_params_, bayes_search.best_score_

def tune_mlp(X, y):
    param_space = {
        'hidden_layer_sizes': Categorical([50, 100, 150]),
        'activation': Categorical(['relu', 'tanh']),
        'solver': Categorical(['adam', 'sgd']),
        'alpha': Real(1e-5, 1e-2, prior='log-uniform'),
        'learning_rate': Categorical(['constant', 'adaptive'])
    }

    model = MLPClassifier(max_iter=5000)

    try:
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=10,
            cv=StratifiedKFold(n_splits=5),
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        bayes_search.fit(X, y)

        return bayes_search.best_params_, bayes_search.best_score_

    except Exception as e:
        print("Error during optimization!")
        return None, None

# Main Script
if __name__ == "__main__":
    # Load data
    table_name = 'keystrokes'
    X,y = load_data(session, table_name)
    
    # Load existing scaler
    print("Loading existing scaler...")
    scaler = joblib.load(scaler_path)
    
    # Standardize features
    print("Scaling the data...")
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    results = {}
    # Neural Network
    print("Tuning Neural Network...")
    best_params, best_score = tune_mlp(X_scaled, y)
    results['Neural Network'] = {'params': best_params, 'score': best_score}

    # Logistic Regression
    print("Tuning Logistic Regression...")
    best_params, best_score = tune_logistic_regression(X_scaled, y)
    results['Logistic Regression'] = {'params': best_params, 'score': best_score}
    
    # Random Forest
    print("Tuning Random Forest...")
    best_params, best_score = tune_random_forest(X_scaled, y)
    results['Random Forest'] = {'params': best_params, 'score': best_score}
    
    # Support Vector Machine
    print("Tuning Support Vector Machine...")
    best_params, best_score = tune_svc(X_scaled, y)
    results['Support Vector Machine'] = {'params': best_params, 'score': best_score}
    
    # Gradient Boosting
    print("Tuning Gradient Boosting...")
    best_params, best_score = tune_gradient_boosting(X_scaled, y)
    results['Gradient Boosting'] = {'params': best_params, 'score': best_score}
    
    # Save Results
    print("Saving results...")
    with open('model_tuning_results_without_weights.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Optimization complete. Results saved to model_tuning_results_without_weights.json.")
