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

# Path to the database
db_path = r'D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db'
scaler_path = r'D:\THESIS\MobileApp\Flask_server\models\scaler_username.joblib'

# Create an SQLAlchemy engine to connect to the SQLite database
engine = create_engine(f'sqlite:///{db_path}', echo=True)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Feature columns and weights as defined in your training script
expected_columns = [
            'press_press_interval_mean',
            'release_interval_mean',
            'hold_time_mean',
            'press_press_interval_variance',
            'release_interval_variance',
            'hold_time_variance',
            'error_rate',
            'total_typing_time',
            'typing_speed_cps'
        ]

# Define weights for each feature
feature_weights = {
    'press_press_interval_mean': 1.4,  # Slightly less important than username press-press interval mean
    'release_interval_mean': 1.1,  # Less important compared to other features
    'hold_time_mean': 1.4,  # Relatively important feature in most models
    'press_press_interval_variance': 1.5,  # High importance due to significant variance in keystroke dynamics
    'release_interval_variance': 1.3,  # Moderately important
    'hold_time_variance': 1.3,  # Moderately important
    'error_rate': 1.0,  # Slightly less important
    'total_typing_time': 1.2,  # Important in some models like SVM
    'typing_speed_cps': 1.4,  # Important across models
}

expected_columns_with_username = [
    'press_press_interval_mean',
    'release_interval_mean',
    'hold_time_mean',
    'press_press_interval_variance',
    'release_interval_variance',
    'hold_time_variance',
    'error_rate',
    'total_typing_time',
    'typing_speed_cps',
    'username_press_press_interval_mean',
    'username_release_interval_mean',
    'username_hold_time_mean',
    'username_press_press_interval_variance',
    'username_release_interval_variance',
    'username_hold_time_variance',
    'username_error_rate',
    'username_total_typing_time',
    'username_typing_speed_cps'
]

feature_weights_with_username = {
    'press_press_interval_mean': 1.4,
    'release_interval_mean': 1.1,
    'hold_time_mean': 1.3,
    'press_press_interval_variance': 1.5,
    'release_interval_variance': 1.3,
    'hold_time_variance': 1.3,
    'error_rate': 1.0,
    'total_typing_time': 1.2,
    'typing_speed_cps': 1.4,
    'username_press_press_interval_mean': 1.3,
    'username_release_interval_mean': 1.0,
    'username_hold_time_mean': 1.0,
    'username_press_press_interval_variance': 1.3,
    'username_release_interval_variance': 1.1,
    'username_hold_time_variance': 1.1,
    'username_error_rate': 0.8,
    'username_total_typing_time': 0.9,
    'username_typing_speed_cps': 1.1
}

# Modify the load_data function to handle both sets of features
def load_data_with_username(session, table_name, with_username=False):
    query = text(f"SELECT * FROM {table_name}")
    result = session.execute(query)
    data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if with_username:
        # Use both the original and username-specific features
        X = data[expected_columns_with_username].copy()
    else:
        # Use only the original features
        X = data[expected_columns].copy()
        
    y = data['user_id']  # Replace with the actual target column name

    # Apply weights to the features
    feature_weights_dict = feature_weights_with_username if with_username else feature_weights
    for feature in X.columns:
        X[feature] *= feature_weights_dict.get(feature, 1.0)  # Apply weight or default to 1.0

    return X, y
# Load Data from Database Using SQLAlchemy Session
def load_data(session, table_name):
    query = text(f"SELECT * FROM {table_name}")
    result = session.execute(query)
    data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    # Ensure consistent feature selection
    X = data[expected_columns].copy()
    y = data['user_id']  # Replace with the actual target column name

    # Apply weights to the features
    for feature in expected_columns:
        X[feature] *= feature_weights[feature]

    return X, y

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
    print("Starting Neural Network Tuning...")
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"Unique labels in y: {np.unique(y)}")

    # Refined parameter space
    param_space = {
        'hidden_layer_sizes': Categorical([50, 100, 150]),  # Explicit Categorical
        'activation': Categorical(['relu', 'tanh']),
        'solver': Categorical(['adam', 'sgd']),
        'alpha': Real(1e-5, 1e-2, prior='log-uniform'),  # Fix 'identity'
        'learning_rate': Categorical(['constant', 'adaptive'])
    }

    model = MLPClassifier(max_iter=5000)

    try:
        print("Initializing BayesSearchCV...")
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

        print("Starting optimization...")
        bayes_search.fit(X, y)

        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best score: {bayes_search.best_score_}")

        return bayes_search.best_params_, bayes_search.best_score_

    except Exception as e:
        print("Error during optimization!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Current search space:")
        print(param_space)

        # Log sampled parameters causing the issue
        try:
            print(f"Sampled parameters: {bayes_search.cv_results_['params']}")
        except Exception:
            print("Could not retrieve cv_results_.")
        return None, None

if __name__ == "__main__":
    # Load data without username
    table_name = 'preprocessed_keystroke_data'
    
    # Standard optimization (without username features)
    print("Loading data without username features...")
    X, y = load_data(session, table_name)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    joblib.dump(StandardScaler(), scaler_path)

    results = {}
    # Neural Network (without username)
    print("Tuning Neural Network without username...")
    best_params, best_score = tune_mlp(X_scaled, y)
    results['Neural Network'] = {'params': best_params, 'score': best_score}

    # Logistic Regression (without username)
    print("Tuning Logistic Regression without username...")
    best_params, best_score = tune_logistic_regression(X_scaled, y)
    results['Logistic Regression'] = {'params': best_params, 'score': best_score}
    
    # Random Forest (without username)
    print("Tuning Random Forest without username...")
    best_params, best_score = tune_random_forest(X_scaled, y)
    results['Random Forest'] = {'params': best_params, 'score': best_score}

    # Support Vector Machine (without username)
    print("Tuning Support Vector Machine without username...")
    best_params, best_score = tune_svc(X_scaled, y)
    results['Support Vector Machine'] = {'params': best_params, 'score': best_score}

    # Gradient Boosting (without username)
    print("Tuning Gradient Boosting without username...")
    best_params, best_score = tune_gradient_boosting(X_scaled, y)
    results['Gradient Boosting'] = {'params': best_params, 'score': best_score}

    # Save results for models without username
    print("Saving results for models without username...")
    with open('model_tuning_results_without_username.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Load data with username features
    print("Loading data with username features...")
    X, y = load_data_with_username(session, table_name, with_username=True)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    # Reset results for models with username
    results_with_username = {}

    # Neural Network (with username)
    print("Tuning Neural Network with username...")
    best_params, best_score = tune_mlp(X_scaled, y)
    results_with_username['Neural Network'] = {'params': best_params, 'score': best_score}

    # Logistic Regression (with username)
    print("Tuning Logistic Regression with username...")
    best_params, best_score = tune_logistic_regression(X_scaled, y)
    results_with_username['Logistic Regression'] = {'params': best_params, 'score': best_score}
    
    # Random Forest (with username)
    print("Tuning Random Forest with username...")
    best_params, best_score = tune_random_forest(X_scaled, y)
    results_with_username['Random Forest'] = {'params': best_params, 'score': best_score}

    # Support Vector Machine (with username)
    print("Tuning Support Vector Machine with username...")
    best_params, best_score = tune_svc(X_scaled, y)
    results_with_username['Support Vector Machine'] = {'params': best_params, 'score': best_score}

    # Gradient Boosting (with username)
    print("Tuning Gradient Boosting with username...")
    best_params, best_score = tune_gradient_boosting(X_scaled, y)
    results_with_username['Gradient Boosting'] = {'params': best_params, 'score': best_score}

    # Save results for models with username
    print("Saving results for models with username...")
    with open('model_tuning_results_with_username.json', 'w') as f:
        json.dump(results_with_username, f, indent=4)

    print("Optimization complete. Results saved to model_tuning_results_with_username.json.")