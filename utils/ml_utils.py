import os
import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker
from models import db, PreprocessedKeystrokeData, User
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_model(max_entries=None, use_all_data=False, preprocessed=True):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    print("Train without username")

    try:
        data = pd.read_sql('SELECT * FROM preprocessed_keystroke_data', session.bind)
        data = data[data['user_id'] != 1]  # Exclude admin

        user_groups = data.groupby('user_id')
        min_required = 10

        # If 'use_all_data' is True, use all data without limiting the number of entries per user
        if use_all_data:
            user_groups = data.groupby('user_id')
        else:
            # If 'max_entries' is provided, limit each user to a maximum number of entries
            data = data.groupby('user_id').apply(lambda x: x.head(max_entries)).reset_index(drop=True)
            user_groups = data.groupby('user_id')

        # Check if any user has less than the minimum required entries
        for user_id, group in user_groups:
            if len(group) < min_required:
                return {"error": f"User {user_id} has only {len(group)} entries, need minimum {min_required} for reliable split"}

        test_data = []
        train_data = []

        # Split data into train and test sets
        for user_id, group in user_groups:
            test_size = int(len(group) * 0.2)
            test_samples = group.sample(n=test_size)
            train_samples = group.drop(test_samples.index)

            test_data.append(test_samples)
            train_data.append(train_samples)

        test_set = pd.concat(test_data)
        train_set = pd.concat(train_data)

        train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
        test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

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

        feature_weights = {
            'press_press_interval_mean': 1.4,
            'release_interval_mean': 1.1,
            'hold_time_mean': 1.4,
            'press_press_interval_variance': 1.5,
            'release_interval_variance': 1.3,
            'hold_time_variance': 1.3,
            'error_rate': 1.0,
            'total_typing_time': 1.2,
            'typing_speed_cps': 1.4,
        }

        X_train = train_set[expected_columns].copy()
        y_train = train_set['user_id']
        X_test = test_set[expected_columns].copy()
        y_test = test_set['user_id']

        # Apply feature weights to the training and testing data
        for feature in expected_columns:
            X_train[feature] *= feature_weights[feature]
            X_test[feature] *= feature_weights[feature]

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Save scaler for later use
        scaler_path = os.path.join('models', 'scaler.joblib')
        joblib.dump(scaler, scaler_path)

        models = {
            'Neural Network': MLPClassifier(
                activation="tanh", 
                alpha=0.0012643413272636261, 
                hidden_layer_sizes=15, 
                learning_rate="constant", 
                solver="adam", 
                max_iter=5000
            ),
            'Logistic Regression': LogisticRegression(
                C=78.73105608900913, 
                penalty='l1', 
                solver='saga', 
                max_iter=10000
            ),
            'Random Forest': RandomForestClassifier(
                max_depth=19, 
                min_samples_leaf=2, 
                min_samples_split=5, 
                n_estimators=100, 
                class_weight='balanced'
            ),
            'Support Vector Machine': SVC(
                C=100.0, 
                gamma='scale', 
                kernel='linear', 
                probability=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                learning_rate=0.01, 
                max_depth=2, 
                min_samples_leaf=6, 
                min_samples_split=5, 
                n_estimators=150
            )
        }

        # Create a folder for plots
        os.makedirs('plots', exist_ok=True)

        results = {}
        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            # Create learning curves for each model
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train_scaled, y_train,
                cv=10, n_jobs=-1,
                train_sizes=np.linspace(0.3, 1.0, 10),
                scoring='accuracy'
            )

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
            plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(f'plots/learning_curve_{model_name.lower().replace(" ", "_")}.png')
            plt.close()

            # Train the model and evaluate on the test set
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            test_accuracy = model.score(X_test_scaled, y_test)
            train_accuracy = model.score(X_train_scaled, y_train)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            tn = np.sum(np.diag(cm)) - np.sum(cm, axis=1)  # True Negatives for each user
            fp = np.sum(cm, axis=0) - np.diag(cm)         # False Positives for each user
            fn = np.sum(cm, axis=1) - np.diag(cm)         # False Negatives for each user

            # FAR and FRR
            total_negatives = np.sum(fp) + np.sum(tn)
            total_positives = np.sum(np.diag(cm)) + np.sum(fn)

            far = np.sum(fp) / total_negatives if total_negatives > 0 else 0.0
            frr = np.sum(fn) / total_positives if total_positives > 0 else 0.0

            results[model_name] = {
                "status": "Model trained successfully",
                "test_accuracy": f"{test_accuracy:.4f}",
                "train_accuracy": f"{train_accuracy:.4f}",
                "gap": f"{(train_accuracy - test_accuracy):.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1_score": f"{f1:.4f}",
                "FAR": f"{far:.4f}",
                "FRR": f"{frr:.4f}"
            }

            # Save the trained model
            model_path = os.path.join('models', f'{model_name.replace(" ", "_").lower()}.joblib')
            joblib.dump(model, model_path)

        print("\nOverfitting Analysis:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"Training Accuracy: {metrics['train_accuracy']}")
            print(f"Testing Accuracy: {metrics['test_accuracy']}")
            print(f"Gap (Training-Testing): {metrics['gap']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"F1 Score: {metrics['f1_score']}")
            print(f"FAR: {metrics['FAR']}")
            print(f"FRR: {metrics['FRR']}")

        importance_df = extract_feature_importances(models, X_train_scaled, model_names=['Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Gradient Boosting'])
        print(importance_df)

        return results

    except Exception as e:
        return {"error": str(e)}

    finally:
        session.close()
def train_model_with_username(max_entries=None, use_all_data=False, preprocessed=True):
    Session = sessionmaker(bind=db.engine)
    session = Session()
    print("Train with username")

    try:
        data = pd.read_sql('SELECT * FROM preprocessed_keystroke_data', session.bind)
        data = data[data['user_id'] != 1]  # Exclude admin
        
        user_groups = data.groupby('user_id')
        # If 'use_all_data' is True, use all data without limiting the number of entries per user
        if use_all_data:
            user_groups = data.groupby('user_id')
        else:
            # If 'max_entries' is provided, limit each user to a maximum number of entries
            data = data.groupby('user_id').apply(lambda x: x.head(max_entries)).reset_index(drop=True)
            user_groups = data.groupby('user_id')
        min_required = 10
        
        for user_id, group in user_groups:
            if len(group) < min_required:
                return {"error": f"User {user_id} has only {len(group)} entries, need minimum {min_required} for reliable split"}

        test_data = []
        train_data = []
        
        for user_id, group in user_groups:
            test_size = int(len(group) * 0.2)
            test_samples = group.sample(n=test_size)
            train_samples = group.drop(test_samples.index)
            
            test_data.append(test_samples)
            train_data.append(train_samples)
        
        test_set = pd.concat(test_data)
        train_set = pd.concat(train_data)
        train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
        
        test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

        expected_columns = [
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
        
        feature_weights = {
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
        
        X_train = train_set[expected_columns].copy()
        y_train = train_set['user_id']
        X_test = test_set[expected_columns].copy()
        y_test = test_set['user_id']
        
        for feature in expected_columns:
            X_train[feature] *= feature_weights[feature]
            X_test[feature] *= feature_weights[feature]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        scaler_path = os.path.join('models', 'scaler_username.joblib')
        joblib.dump(scaler, scaler_path)
        
        
        models = {
    'Neural Network': MLPClassifier(
    activation="tanh",
    alpha=0.1,  # Stronger regularization
    hidden_layer_sizes=10,  # Reduce complexity
    learning_rate="adaptive",
    solver="adam",
    max_iter=5000
),
    'Logistic Regression': LogisticRegression(
    C=5.0,  # Lower regularization strength than 27.89 to reduce overfitting
    penalty='l2',  # Switch to L2 for smooth regularization
    solver='lbfgs',  # Efficient solver for L2 regularization
    max_iter=5000
),
    'Random Forest': RandomForestClassifier(
    max_depth=10,  # Shallower trees for better generalization
    min_samples_leaf=5,  # Require more samples per leaf to avoid overfitting
    min_samples_split=10,  # Require more samples per split for balanced trees
    n_estimators=200,  # More estimators for stable performance
    max_features='sqrt',  # Consider fewer features per split to reduce complexity
    class_weight='balanced'  # Handle imbalanced data better
),
    'Support Vector Machine': SVC(
    C=1.0,  # Lower regularization parameter to reduce overfitting
    gamma='scale',  # Automatically scales gamma for better feature distribution
    kernel='rbf',  # Use radial basis function for non-linear boundaries
    probability=True  # Enables probability outputs
),
    'Gradient Boosting': GradientBoostingClassifier(
        learning_rate=0.01, 
                max_depth=2, 
                min_samples_leaf=6, 
                min_samples_split=5, 
                n_estimators=150
    )
}

   
        os.makedirs('plots', exist_ok=True)
        
        results = {}
        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            # Create learning curves for each model
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train_scaled, y_train,
                cv=10, n_jobs=-1,
                train_sizes=np.linspace(0.3, 1.0, 10),
                scoring='accuracy'
            )

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
            plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(f'plots/learning_curve_with_username{model_name.lower().replace(" ", "_")}.png')
            plt.close()

            # Train the model and evaluate on the test set
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            test_accuracy = model.score(X_test_scaled, y_test)
            train_accuracy = model.score(X_train_scaled, y_train)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            tn = np.sum(np.diag(cm)) - np.sum(cm, axis=1)  # True Negatives for each user
            fp = np.sum(cm, axis=0) - np.diag(cm)         # False Positives for each user
            fn = np.sum(cm, axis=1) - np.diag(cm)         # False Negatives for each user

            # FAR and FRR
            total_negatives = np.sum(fp) + np.sum(tn)
            total_positives = np.sum(np.diag(cm)) + np.sum(fn)

            far = np.sum(fp) / total_negatives if total_negatives > 0 else 0.0
            frr = np.sum(fn) / total_positives if total_positives > 0 else 0.0

            results[model_name] = {
                "status": "Model trained successfully",
                "test_accuracy": f"{test_accuracy:.4f}",
                "train_accuracy": f"{train_accuracy:.4f}",
                "gap": f"{(train_accuracy - test_accuracy):.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1_score": f"{f1:.4f}",
                "FAR": f"{far:.4f}",
                "FRR": f"{frr:.4f}"
            }

            # Save the trained model
            model_path = os.path.join('models', f'{model_name.replace(" ", "_").lower()}.joblib')
            joblib.dump(model, model_path)

        print("\nOverfitting Analysis:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"Training Accuracy: {metrics['train_accuracy']}")
            print(f"Testing Accuracy: {metrics['test_accuracy']}")
            print(f"Gap (Training-Testing): {metrics['gap']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"F1 Score: {metrics['f1_score']}")
            print(f"FAR: {metrics['FAR']}")
            print(f"FRR: {metrics['FRR']}")

        importance_df = extract_feature_importances(models, X_train_scaled, model_names=['Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Gradient Boosting'])
        print(importance_df)

        return results

    except Exception as e:
        return {"error": str(e)}

    finally:
        session.close()
def extract_feature_importances(models, X, model_names=None):
    importances = {}
    
    for model_name, model in models.items():
        if model_name not in model_names:
            continue
        
        if isinstance(model, LogisticRegression):
            # For Logistic Regression, use the absolute value of the coefficients
            importances[model_name] = np.abs(model.coef_[0])
            
        elif isinstance(model, RandomForestClassifier):
            # For Random Forests, use the feature_importances_ attribute
            importances[model_name] = model.feature_importances_
            
        elif isinstance(model, SVC):
            # For SVM with linear kernel, use the absolute values of the coefficients
            if model.kernel == 'linear':
                importances[model_name] = np.abs(model.coef_[0])
            else:
                importances[model_name] = np.zeros(X.shape[1])  # No importance for non-linear kernels
                
        elif isinstance(model, GradientBoostingClassifier):
            # For Gradient Boosting, use the feature_importances_ attribute
            importances[model_name] = model.feature_importances_
            
        
            
        else:
            importances[model_name] = np.zeros(X.shape[1])  # Default to zeros if model is unknown
    
    # Convert to a DataFrame for easier inspection
    importance_df = pd.DataFrame(importances, index=X.columns)
    return importance_df