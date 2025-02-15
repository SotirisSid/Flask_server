import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.preprocessing import StandardScaler
db_path = r'D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db'

def grid_search_weight_optimization(max_entries=None, use_all_data=False, preprocessed=True):
    engine = create_engine(f'sqlite:///{db_path}', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    print("Grid search for weight optimization")

    try:
        # Load preprocessed data
        data = pd.read_sql('SELECT * FROM preprocessed_keystroke_data', session.bind)

        # Exclude admin user (user_id == 1)
        data = data[data['user_id'] != 1]
        
        # Check if max_entries is provided and if each user has enough data
        if max_entries is not None:
            for user_id, group in data.groupby('user_id'):
                if len(group) < max_entries:
                    return {"error": f"User {user_id} has only {len(group)} entries, which is less than the required {max_entries} entries."}

        # If use_all_data is False and max_entries is provided, limit the data for each user
        if not use_all_data and max_entries is not None:
            data = data.groupby('user_id').head(max_entries)

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
        
        # Define the grid for weight optimization
        weight_grid = {
            'press_press_interval_mean': [1.0, 1.5, 2.0],
            'release_interval_mean': [1.0, 1.2, 1.5],
            'hold_time_mean': [1.0, 1.2, 1.5],
            'press_press_interval_variance': [1.0, 1.5, 2.0],
            'release_interval_variance': [1.0, 1.3, 1.5],
            'hold_time_variance': [1.0, 1.3, 1.5],
            'backspace_count': [0.5, 1.0, 1.5],
            'error_rate': [0.5, 1.0, 1.5],
            'total_typing_time': [1.0, 1.1, 1.5],
            'typing_speed_cps': [1.0, 1.4, 1.6]
        }

        # Flatten the grid of weights for all features
        from sklearn.model_selection import ParameterGrid
        weight_combinations = list(ParameterGrid(weight_grid))
        
        # Extract features and target variable
        X = data[expected_columns].copy()
        y = data['user_id']

        # Load the pre-existing scaler
        scaler_path = os.path.join('models', 'scaler.joblib')
        scaler = joblib.load(scaler_path)

        # Standardize features using the loaded scaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(max_iter=10000, 
                                                    C=46.610688373989866, 
                                                    penalty='l1', 
                                                    solver='saga'),
            'Random Forest': RandomForestClassifier(n_estimators=197, 
                                                    max_depth=26, 
                                                    min_samples_leaf=1, 
                                                    min_samples_split=2, 
                                                    class_weight='balanced'),
            'Support Vector Machine': SVC(probability=True, 
                                        C=59.81009454859274, 
                                        gamma='auto', 
                                        kernel='linear'),
            'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.2, 
                                                            max_depth=3, 
                                                            min_samples_leaf=1, 
                                                            min_samples_split=2, 
                                                            n_estimators=200),
            'Neural Network': MLPClassifier(max_iter=5000, 
                                            activation='tanh', 
                                            alpha=0.0012643413272636261, 
                                            hidden_layer_sizes=150, 
                                            learning_rate='constant', 
                                            solver='adam')
        }

        # Perform grid search over weight combinations
        best_results = {}
        for model_name, model in models.items():
            print(f"Running grid search for model: {model_name}")
            best_score = -1
            best_weights = None

            for weights in weight_combinations:
                # Apply weights to the features
                X_weighted = X_scaled.copy()
                for feature in expected_columns:
                    X_weighted[feature] *= weights[feature]

                # Train the model
                model.fit(X_weighted, y)

                # Evaluate the model performance
                y_pred = model.predict(X_weighted)
                score = accuracy_score(y, y_pred)

                if score > best_score:
                    best_score = score
                    best_weights = weights
            
            # Store best results
            best_results[model_name] = {
                'best_score': best_score,
                'best_weights': best_weights
            }

        return best_results

    except Exception as e:
        return {"error": str(e)}

    finally:
        session.close()


if __name__ == "__main__":
    # Any other optimizations or setup
    result = grid_search_weight_optimization(max_entries=15, use_all_data=False, preprocessed=True)
    print("Grid Search Weight Optimization Results:", result)
