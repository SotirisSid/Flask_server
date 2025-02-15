import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker
from models import db
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils.preprocess_keystrokes import process_single_keystroke_data

def plot_learning_curves(model, model_name, X, y):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curves for {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'plots/learning_curve_{model_name}.png')
        plt.close()

def train_raw_model(max_entries=None, use_all_data=False, max_feature_length=100, chunk_size=5000):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    print("Training without username data")

    try:
        # Check for models and plots directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Initialize variables
        features = []
        labels = []
        offset = 0

        while True:
            # Load data in chunks
            query = f"SELECT * FROM keystrokes WHERE user_id != 1 LIMIT {chunk_size} OFFSET {offset}"
            chunk = pd.read_sql_query(query, session.bind)

            if chunk.empty:
                break

            # Group data by user_id to apply max_entries logic
            grouped_chunk = chunk.groupby('user_id')

            for user_id, user_data in grouped_chunk:
                # Limit data per user if max_entries is provided
                if max_entries:
                    user_data = user_data.head(max_entries)  # Use only the first 'max_entries' rows

                for _, row in user_data.iterrows():
                    try:
                        # Parse stringified lists into actual Python lists
                        press_press_intervals = eval(row['press_press_intervals'])
                        release_press_intervals = eval(row['release_press_intervals'])
                        hold_times = eval(row['hold_times'])

                        # Check if the sequence is excessively long (junk)
                        total_length = len(press_press_intervals) + len(release_press_intervals) + len(hold_times)
                        if total_length > max_feature_length:
                            print(f"Skipping user {user_id} due to excessive keystroke length.")
                            continue

                        # Pad feature vectors to the fixed max password length
                        max_password_len = 18
                        press_press_intervals = press_press_intervals[:max_password_len] + [0] * (max_password_len - len(press_press_intervals))
                        release_press_intervals = release_press_intervals[:max_password_len] + [0] * (max_password_len - len(release_press_intervals))
                        hold_times = hold_times[:max_password_len] + [0] * (max_password_len - len(hold_times))

                        # Calculate means and variances for the password intervals
                        press_press_intervals_mean = np.mean(press_press_intervals)
                        press_press_intervals_variance = np.var(press_press_intervals)
                        release_press_intervals_mean = np.mean(release_press_intervals)
                        release_press_intervals_variance = np.var(release_press_intervals)
                        hold_times_mean = np.mean(hold_times)
                        hold_times_variance = np.var(hold_times)

                        # Construct the feature vector
                        feature_vector = (
                            press_press_intervals +
                            [press_press_intervals_variance]+
                            
                            [row['typing_speed']]
                            
                            
                        )
                        
                        # Convert feature vector values to float and handle NaN/inf
                        feature_vector = [float(x) if pd.notna(x) and np.isfinite(x) else 0.0 for x in feature_vector]
                        features.append(feature_vector)
                        labels.append(user_id)
                    except Exception as e:
                        print(f"Error processing row {row['id']}: {str(e)}")

            offset += chunk_size

        # Convert features and labels to DataFrame
        X = pd.DataFrame(features)
        y = pd.Series(labels)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X))
        print(f"Shape of training feature matrix: {X_scaled.shape}")

        # Save the scaler
        scaler_path = os.path.join('models', 'raw_scaler.joblib')
        joblib.dump(scaler, scaler_path)

        # Define models (as previously defined)
        models = {
    'logistic_regression': LogisticRegression(
        C=1.0,  # Regularization strength
        penalty="l2",  # L2 regularization
        solver="saga",
        max_iter=1000
    ),
    'random_forest': RandomForestClassifier(
        max_depth=5,  # Depth of the trees
        n_estimators=50  # Number of trees
    ),
    'svm': SVC(
        C=1.0,  # Regularization strength
        kernel='rbf',  # Non-linear kernel
        probability=True
    ),
    'gradient_boosting': GradientBoostingClassifier(
        learning_rate=0.1,  # Learning rate
        n_estimators=100,  # Number of boosting stages
        max_depth=3  # Depth of the individual trees
    ),
    'neural_network': MLPClassifier(
        hidden_layer_sizes=(100,),  # Single hidden layer with 100 units
        activation="relu",  # Activation function
        max_iter=1000
    )
}

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        def train_and_evaluate_model(model_name, model):
            try:
                # Add cross-validation scores
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                
                # Fit the model on the full training data
                model.fit(X_train, y_train)
                
                # Get predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate accuracies
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                cv_accuracy = np.mean(cv_scores)
                
                print(f"{model_name} CV scores: {cv_scores}")
                print(f"{model_name} Mean CV accuracy: {cv_accuracy:.4f}")
                
                # Save the model
                model_path = os.path.join('models', f'{model_name}Raw.joblib')
                joblib.dump(model, model_path)
                
                return model_name, train_accuracy, test_accuracy, cv_accuracy
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                return model_name, None, None

        # Train models in parallel
        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_model)(model_name, model) for model_name, model in models.items()
        )

        # Display results
        for model_name, train_accuracy, test_accuracy, cv_accuracy in results:
            if train_accuracy is not None:
                print(f"{model_name}:")
                print(f"Train Accuracy = {train_accuracy:.4f}")
                print(f"Test Accuracy = {test_accuracy:.4f}") 
                print(f"CV Accuracy = {cv_accuracy:.4f}")
                print("---")

        # After training each model
        for model_name, model in models.items():
            plot_learning_curves(model, model_name, X_scaled, y)        

        return {"message": "Models trained successfully"}

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {"error": str(e)}


def train_with_username_data_raw(max_entries=None, use_all_data=False, max_feature_length=100, chunk_size=5000):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    print("Training with username-specific data")

    try:
        # Check for models and plots directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Initialize variables
        features = []
        labels = []
        offset = 0

        while True:
            # Load data in chunks
            query = f"SELECT * FROM keystrokes WHERE user_id != 1 LIMIT {chunk_size} OFFSET {offset}"
            chunk = pd.read_sql_query(query, session.bind)

            if chunk.empty:
                break

            # Group data by user_id to apply max_entries logic
            grouped_chunk = chunk.groupby('user_id')

            for user_id, user_data in grouped_chunk:
                if use_all_data or not max_entries:
                    # If use_all_data is True, or max_entries is not specified, use all data for the user
                    user_data_to_use = user_data
                else:
                    # Limit data per user if max_entries is provided
                    user_data_to_use = user_data.head(max_entries)

                for _, row in user_data_to_use.iterrows():
                    try:
                        # Parse stringified lists into actual Python lists
                        press_press_intervals = eval(row['press_press_intervals'])
                        release_press_intervals = eval(row['release_press_intervals'])
                        hold_times = eval(row['hold_times'])
                        username_press_press_intervals = eval(row['username_press_press_intervals'])
                        username_release_press_intervals = eval(row['username_release_press_intervals'])
                        username_hold_times = eval(row['username_hold_times'])

                        # Check if the sequence is excessively long (junk)
                        total_length = len(press_press_intervals) + len(release_press_intervals) + len(hold_times)
                        if total_length > max_feature_length:
                            print(f"Skipping user {row['user_id']} due to excessive keystroke length.")
                            continue

                        # Pad feature vectors to the fixed max password length
                        max_password_len = 18
                        press_press_intervals = press_press_intervals[:max_password_len] + [0] * (max_password_len - len(press_press_intervals))
                        release_press_intervals = release_press_intervals[:max_password_len] + [0] * (max_password_len - len(release_press_intervals))
                        hold_times = hold_times[:max_password_len] + [0] * (max_password_len - len(hold_times))
                        username_press_press_intervals = username_press_press_intervals[:max_password_len] + [0] * (max_password_len - len(username_press_press_intervals))
                        username_release_press_intervals = username_release_press_intervals[:max_password_len] + [0] * (max_password_len - len(username_release_press_intervals))
                        username_hold_times = username_hold_times[:max_password_len] + [0] * (max_password_len - len(username_hold_times))
                        username_processed_data = process_single_keystroke_data(
                            row['user_id'],
                            username_press_press_intervals,
                            username_release_press_intervals,
                            username_hold_times,
                            row['total_typing_time'],
                            row['typing_speed'],
                            row['backspace_count'],
                            row['error_rate']
                        )

                        password_processed_data = process_single_keystroke_data(
                            row['user_id'],
                            press_press_intervals,
                            release_press_intervals,
                            hold_times,
                            row['total_typing_time'],
                            row['typing_speed'],
                            row['backspace_count'],
                            row['error_rate']
                        )

                        # Construct the feature vector
                        feature_vector = (
                            press_press_intervals +
                            release_press_intervals +
                            hold_times +
                            [
                            password_processed_data['press_press_interval_mean'],
                            password_processed_data['press_press_interval_variance'],
                            password_processed_data['hold_time_mean'],
                            password_processed_data['hold_time_variance'],
                            password_processed_data['release_interval_variance']
                            ] +
                            username_press_press_intervals +
                            username_release_press_intervals +
                            username_hold_times +
                            [
                            username_processed_data['press_press_interval_mean'],
                            username_processed_data['press_press_interval_variance'],
                            username_processed_data['hold_time_mean'],
                            username_processed_data['hold_time_variance'],
                            username_processed_data['release_interval_variance']
                            ] +
                            [
                                row['total_typing_time'],
                                row['typing_speed'],
                                row['error_rate'],
                                row['press_to_release_ratio_mean'],
                                row['username_total_typing_time'],
                                row['username_typing_speed_cps'],
                                row['error_rate_username']
                            ]
                        )

                        # Convert feature vector values to float and handle NaN/inf
                        feature_vector = [float(x) if pd.notna(x) and np.isfinite(x) else 0.0 for x in feature_vector]
                        features.append(feature_vector)
                        labels.append(row['user_id'])
                    except Exception as e:
                        print(f"Error processing row {row['id']}: {str(e)}")

            offset += chunk_size

        # Convert features and labels to DataFrame
        X = pd.DataFrame(features)
        y = pd.Series(labels)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X))
        print(f"Shape of training feature matrix: {X_scaled.shape}")

        # Save the scaler
        scaler_path = os.path.join('models', 'Rawusername_scaler.joblib')
        joblib.dump(scaler, scaler_path)

        # Define models
        models = {
    'logistic_regression': LogisticRegression(
        C=0.1,  # Stronger regularization
        penalty="l2",  # L2 regularization
        solver="saga",
        max_iter=7000
    ),
    'random_forest': RandomForestClassifier(
        max_depth=8,  # Reduce depth
        min_samples_leaf=4,  # Increase min samples
        min_samples_split=10,
        n_estimators=100
    ),
            'svm': SVC(
                C=1.0,  # Reduced from 100.0 to decrease model complexity
    gamma='auto',  # Let sklearn choose optimal gamma
    kernel='rbf',  # Non-linear kernel for better generalization
    probability=True,
    class_weight='balanced'  # Handle class imbalance
            ),
            'gradient_boosting': GradientBoostingClassifier(
    learning_rate=0.05,  # Reduced learning rate for better generalization
    max_depth=2,  # Reduced from 3
    min_samples_leaf=4,  # Increased from 2
    min_samples_split=15,  # Increased from 10
    n_estimators=150,  # Reduced from 200
    subsample=0.8,  # Add random sampling
    validation_fraction=0.2,  # Add validation monitoring
    n_iter_no_change=10  # Early stopping
),
            'neural_network': MLPClassifier(
                activation="tanh",
                alpha=0.01,
                hidden_layer_sizes=150,
                learning_rate="constant",
                early_stopping=True,
                solver="adam",
                max_iter=1000
            )
        }

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        def train_and_evaluate_model(model_name, model):
            try:
                # Fit the model
                model.fit(X_train, y_train)

                # Predict on training and testing data
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate accuracy
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                # Save the trained model
                model_path = os.path.join('models', f'{model_name}RawUsername.joblib')
                joblib.dump(model, model_path)

                return model_name, train_accuracy, test_accuracy
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                return model_name, None, None

        # Train models in parallel
        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_model)(model_name, model) for model_name, model in models.items()
        )

        # Display results
        for model_name, train_accuracy, test_accuracy in results:
            if train_accuracy is not None:
                print(f"{model_name}: Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

        return {"message": "Models trained successfully with username data"}

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {"error": str(e)}
    


    
    
    