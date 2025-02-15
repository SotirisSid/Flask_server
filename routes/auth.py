import json
from flask import Blueprint, request, jsonify
from flask import current_app
import jwt
import datetime
import numpy as np
from werkzeug.security import check_password_hash
from utils.preprocess_keystrokes import process_single_keystroke_data  # Import the new function

from models import Keystroke, db, User, PreprocessedKeystrokeData,AuthenticationEvaluation  
import os
import joblib  # Import joblib for loading models
import pandas as pd  # Import pandas
import globals
from utils.calculate_features import calculate_keystroke_features  # Import the feature calculation function

# This script is used to authenticate a user and generate a JWT token for the user
auth_bp = Blueprint('auth', __name__)

def generate_token(username):
    """
    Generate a JWT token for the authenticated user.
    """
    try:
        token = jwt.encode({
            'sub': username,
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=10)
        }, current_app.config['SECRET_KEY'], algorithm='HS256')
        return token
    except Exception as e:
        print(f"Error generating token: {e}")  # Log the error
        return None

@auth_bp.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        # Define threshold
        threshold = 0.4  # Test threshold value

        # Retrieve the JSON data from the request
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON'}), 400

        # Extract required fields
        username = data.get('username')
        password = data.get('password')
        press_times = data.get('key_press_times', '')
        release_times = data.get('key_release_times', '')
        backspace_count = data.get('backspace_count', 0)
        error_rate = data.get('error_rate', 0.0)
        ## Add username keypress and keyrelease
        username_keypress = data.get('key_press_times_username', '')
        username_keyrelease = data.get('key_release_times_username', '')
        username_backspace = data.get('backspace_count_username', 0)
        user_status= data.get("user_status")
        
       

        # Validate and process press_times and release_times
        if isinstance(press_times, str):
            press_times = [float(x) for x in press_times.split(',')]
            username_keypress = [float(x) for x in username_keypress.split(',')]

        if isinstance(release_times, str):
            release_times = [float(x) for x in release_times.split(',')]
            username_keyrelease = [float(x) for x in username_keyrelease.split(',')]
        if not isinstance(username_keypress, list) or not isinstance(username_keyrelease, list):
            return jsonify({'error': 'username_keypress and username_keyrelease must be lists'}), 400

        if not isinstance(press_times, list) or not isinstance(release_times, list):
            return jsonify({'error': 'press_times and release_times must be lists'}), 400
        username_error_rate = (username_backspace /
                           (int(len(username_keypress)) + username_backspace))*100
        # Calculate keystroke features
        keystroke_features = calculate_keystroke_features(press_times, release_times)
        username_keystroke_features = calculate_keystroke_features(username_keypress, username_keyrelease)
        processed_data = process_single_keystroke_data(
            username,
            keystroke_features['press_press_intervals'],
            keystroke_features['release_press_intervals'],
            keystroke_features['hold_times'],
            keystroke_features['total_typing_time'],
            keystroke_features['typing_speed_cps'],
            backspace_count,
            error_rate
        )

        # Find the user in the database (move this up)
        user = User.query.filter_by(username=username).first()
        
        if user is None:
            return jsonify({
                'authenticated': False,
                'message': 'Invalid username or password.'
            }), 401
        role = user.role
        ##skip the model prediction if the user is admin
        
        if role=="admin":
            
            return jsonify({
            'authenticated': True,
            'token': generate_token(username),
            'predictions': ["no predictions for admin"],
            'role': role
        }), 200

        # Process username keystroke data (moved after user query)
        username_preprocessed_data = process_single_keystroke_data(
            user_id=user.id,
            press_press_intervals=username_keystroke_features["press_press_intervals"],  # Remove "username_" prefix
            release_press_intervals=username_keystroke_features["release_press_intervals"],  # Remove "username_" prefix
            hold_times=username_keystroke_features["hold_times"],  # Remove "username_" prefix
            total_typing_time=username_keystroke_features["total_typing_time"],  # Remove "username_" prefix
            typing_speed_cps=username_keystroke_features["typing_speed_cps"],  # Remove "username_" prefix
            backspace_count=username_backspace,
            error_rate=username_error_rate
        )

        if processed_data is None or username_preprocessed_data is None:
            return jsonify({'error': 'Error processing keystroke data.'}), 500

        # Prepare the feature vector for model prediction
        feature_vector = [
            processed_data['press_press_interval_mean'],
            processed_data['release_interval_mean'],
            processed_data['hold_time_mean'],
            processed_data['press_press_interval_variance'],
            processed_data['release_interval_variance'],
            processed_data['hold_time_variance'],
            
            error_rate,
            processed_data['total_typing_time'],
            processed_data['typing_speed_cps']
        ]
        extended_feature_vector = [
            processed_data['press_press_interval_mean'],
            processed_data['release_interval_mean'],
            processed_data['hold_time_mean'],
            processed_data['press_press_interval_variance'],
            processed_data['release_interval_variance'],
            processed_data['hold_time_variance'],
            
            error_rate,
            processed_data['total_typing_time'],
            processed_data['typing_speed_cps'],
            # Username features
            username_preprocessed_data['press_press_interval_mean'],
            username_preprocessed_data['release_interval_mean'],
            username_preprocessed_data['hold_time_mean'],
            username_preprocessed_data['press_press_interval_variance'],
            username_preprocessed_data['release_interval_variance'],
            username_preprocessed_data['hold_time_variance'],
            
            username_error_rate,
            username_preprocessed_data['total_typing_time'],
            username_preprocessed_data['typing_speed_cps']
        ]

        # Load the scaler
        scaler_path = os.path.join('models', 'scaler.joblib')
        scaler = joblib.load(scaler_path)
        #load the scaler for extended features
        scaler_path_extended = os.path.join('models', 'scaler_username.joblib')
        scaler_extended = joblib.load(scaler_path_extended)

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
        # Scale the feature vector using a DataFrame
        feature_df = pd.DataFrame([feature_vector], columns=[
            'press_press_interval_mean', 'release_interval_mean',
            'hold_time_mean', 'press_press_interval_variance',
            'release_interval_variance', 'hold_time_variance',
             'error_rate', 'total_typing_time',
            'typing_speed_cps'
        ])
        extended_feature_vector_df = pd.DataFrame([extended_feature_vector], columns=[
            'press_press_interval_mean', 'release_interval_mean',
            'hold_time_mean', 'press_press_interval_variance',
            'release_interval_variance', 'hold_time_variance',
             'error_rate', 'total_typing_time',
            'typing_speed_cps',
            # Username features
            'username_press_press_interval_mean',
            'username_release_interval_mean',
            'username_hold_time_mean',
            'username_press_press_interval_variance',
            'username_release_interval_variance',
            'username_hold_time_variance',
            
            'username_error_rate',
            'username_total_typing_time',
            'username_typing_speed_cps'
        ])
        feature_weights_extended = {
    # Password features
    'press_press_interval_mean': 1.4,  # Close to the original weight
    'release_interval_mean': 1.1,  # Slightly reduced importance
    'hold_time_mean': 1.3,  # Relatively more important in some models
    'press_press_interval_variance': 1.5,  # Remains highly important
    'release_interval_variance': 1.3,  # Moderate importance
    'hold_time_variance': 1.3,  # Moderately important
    
    'error_rate': 1.0,  # Same as previous
    'total_typing_time': 1.2,  # Slightly more important
    'typing_speed_cps': 1.4,  # Still highly important
    
    # Username features (slightly lower weights)
    'username_press_press_interval_mean': 1.3,  # Slightly more important than others
    'username_release_interval_mean': 1.0,  # Least important of the username features
    'username_hold_time_mean': 1.0,  # Moderate importance
    'username_press_press_interval_variance': 1.3,  # Important, but less than password features
    'username_release_interval_variance': 1.1,  # Moderate importance
    'username_hold_time_variance': 1.1,  # Moderate importance
    
    'username_error_rate': 0.8,  # Slightly less important
    'username_total_typing_time': 0.9,  # Important but less than other features
    'username_typing_speed_cps': 1.1  # Slightly lower importance than password typing speed
}
        for column in feature_df.columns:
            if column in feature_weights:
                feature_df[column] = feature_df[column] * feature_weights[column]
        for column in extended_feature_vector_df.columns:
            if column in feature_weights_extended:
                extended_feature_vector_df[column] = extended_feature_vector_df[column] * feature_weights_extended[column]
        # Scaled vectors
        scaled_features = scaler.transform(feature_df)
        scaled_extended_features = scaler_extended.transform(extended_feature_vector_df)

        # Convert scaled features to DataFrame
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_df.columns)
        scaled_extended_features_df = pd.DataFrame(scaled_extended_features, columns=extended_feature_vector_df.columns)
        
        
        
        
        # Check the password
        if not check_password_hash(user.password, password):
            # Password does not match
            return jsonify({
                'authenticated': False,
                'message': 'Invalid username or password.'
            }), 401
        
        

        if  User.query.count() <2:
            return jsonify({
            'authenticated': True,
            'token': generate_token(username),
            'predictions': ["no predictions for single user"],
            'role': role
        }), 200
        keystroke_count = PreprocessedKeystrokeData.query.filter_by(user_id=user.id).count()
        if keystroke_count < 5:
            return jsonify({
            'authenticated': True,
            'token': generate_token(username),
            'predictions': ["no predictions for less than 5 keystrokes"],
            'role': role
        }), 200
        

        # User is authenticated; proceed with keystroke data processing
        if user_status.lower() == 'valid user':
            
            keystroke = Keystroke(
        user_id=user.id,  # Use user.id to link to the user
        press_press_intervals=json.dumps(keystroke_features['press_press_intervals']),  # Store as JSON strings
        release_press_intervals=json.dumps(keystroke_features['release_press_intervals']),  # Store as JSON strings
        hold_times=json.dumps(keystroke_features['hold_times']),  # Store hold times as JSON strings
        total_typing_time=keystroke_features['total_typing_time'],  # Total typing time
        typing_speed=keystroke_features['typing_speed_cps'],  # Typing speed in characters per second
        backspace_count=backspace_count,
        error_rate=error_rate,
        press_to_release_ratio_mean=keystroke_features['press_to_release_ratio_mean'],  # Mean press-to-release ratio
        username_press_press_intervals=json.dumps(username_keystroke_features['press_press_intervals']),
        username_release_press_intervals=json.dumps(username_keystroke_features['release_press_intervals']),
        username_hold_times=json.dumps(username_keystroke_features['hold_times']),
        username_total_typing_time=username_keystroke_features['total_typing_time'],
        username_typing_speed_cps=username_keystroke_features['typing_speed_cps'],
        backspace_count_username=username_backspace,
        error_rate_username=username_error_rate
    )


            
            new_entry = PreprocessedKeystrokeData(
                user_id=user.id,
                press_press_interval_mean=processed_data['press_press_interval_mean'],
                release_interval_mean=processed_data['release_interval_mean'],
                hold_time_mean=processed_data['hold_time_mean'],
                press_press_interval_variance=processed_data['press_press_interval_variance'],
                release_interval_variance=processed_data['release_interval_variance'],
                hold_time_variance=processed_data['hold_time_variance'],
                backspace_count=backspace_count,
                error_rate=error_rate,
                total_typing_time=processed_data['total_typing_time'],
                typing_speed_cps=processed_data['typing_speed_cps'],
                ##username features

                username_press_press_interval_mean=username_preprocessed_data['press_press_interval_mean'],
                username_release_interval_mean=username_preprocessed_data['release_interval_mean'],
                username_hold_time_mean=username_preprocessed_data['hold_time_mean'],
                username_press_press_interval_variance=username_preprocessed_data['press_press_interval_variance'],
                username_release_interval_variance=username_preprocessed_data['release_interval_variance'],
                username_hold_time_variance=username_preprocessed_data['hold_time_variance'],
                username_backspace_count=username_backspace,
                username_error_rate=username_error_rate,
                username_total_typing_time=username_preprocessed_data['total_typing_time'],
                username_typing_speed_cps=username_preprocessed_data['typing_speed_cps']




            )
            db.session.add(keystroke)
            db.session.add(new_entry)
            db.session.commit()

        token = generate_token(username)

        # Check keystroke count
        

        # Load models
        model_names = [
    'logistic_regression.joblib',
    'random_forest.joblib',
    'support_vector_machine.joblib',
    'gradient_boosting.joblib',
    'neural_network.joblib',
    'logistic_regression_with_username.joblib',
    'random_forest_with_username.joblib',
    'support_vector_machine_with_username.joblib',
    'gradient_boosting_with_username.joblib',
    'neural_network_with_username.joblib',
    
]

        prediction_messages = []

        # Initialize model predictions dictionary
        model_predictions = {
    'logistic_regression': None,
    'random_forest': None,
    'support_vector_machine': None,
    'gradient_boosting': None,
    'neural_network': None,
    'logistic_regression_with_username': None,
    'random_forest_with_username': None,
    'support_vector_machine_with_username': None,
    'gradient_boosting_with_username': None,
    'neural_network_with_username': None,
   ## 'svmRaw': None,  # Additional entry for new model
   ## 'logistic_regressionRaw': None,  # Additional entry for new model
   ## 'neural_networkRaw': None,  # Additional entry for new model
   ## 'gradient_boostingRaw': None,  # Additional entry for new model
   ## 'random_forestRaw': None  # Additional entry for new model
}

        # Prediction process
        for model_name in model_names:
            model_path = os.path.join('models', model_name)
            model = joblib.load(model_path)

            # Determine which feature set to use based on model name
            if 'with_username' in model_name:
                features_to_use = scaled_extended_features_df  # Use extended features for models with username
            else:
                features_to_use = scaled_features_df  # Use regular features for other models

            # Make predictions using the correct feature set
            if hasattr(model, 'predict_proba'):
                # Using predict_proba for models that support probability prediction
                proba = model.predict_proba(features_to_use)
                confidence = max(proba[0])
                prediction = model.predict(features_to_use)
                predicted_user_id = int(prediction[0])
                print(f"Predicted User ID: {predicted_user_id}")
                prediction_label = 'valid' if predicted_user_id == user.id and confidence >= threshold else 'intruder'
                message = f"{model_name.replace('.joblib', '')} predicts that you are {prediction_label} with confidence {confidence:.2f}."
            else:
                # Basic prediction for models without predict_proba
                prediction = model.predict(features_to_use)
                predicted_user_id = int(prediction[0])
                prediction_label = 'valid' if predicted_user_id == user.id else 'intruder'
                message = f"{model_name.replace('.joblib', '')} predicts that you are {prediction_label}."

            # Append the prediction message
            prediction_messages.append(message)

            # Save the prediction label to the model_predictions dictionary
            model_key = model_name.replace('.joblib', '')
            model_predictions[model_key] = prediction_label
            """""
        # Prediction for models that use X_scaled (without affecting others)
        X_scaled = extract_features_for_prediction(keystroke_features,processed_data)
        X_scaled_username=extract_features_for_prediction_username(keystroke_features,username_keystroke_features,username_preprocessed_data,processed_data)
        print(X_scaled)
        for raw_model_name in [
            'svmRaw.joblib',
            'logistic_regressionRaw.joblib',
            'neural_networkRaw.joblib',
            'gradient_boostingRaw.joblib',
            'random_forestRaw.joblib',
            'svmRawUsername.joblib',
            'logistic_regressionRawUsername.joblib',
            'random_forestRawUsername.joblib',
            'gradient_boostingRawUsername.joblib',
            'neural_networkRawUsername.joblib',


        ]:
            model_path = os.path.join('models', raw_model_name)
            model = joblib.load(model_path)
            if "RawUsername" in raw_model_name:
                Used_X_scaled = X_scaled_username
            else:
                Used_X_scaled = X_scaled
            # Make predictions using X_scaled (ensuring we don't affect the other models)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(Used_X_scaled)
                confidence = max(proba[0])
                prediction = model.predict(Used_X_scaled)
                predicted_user_id = int(prediction[0])
                print(f"Predicted User ID: {predicted_user_id}")
                print(f"Confidence: {confidence}")
                print("Threshold: ",threshold)

                prediction_label = 'valid' if predicted_user_id == user.id and confidence >= threshold else 'intruder'
                message = f"{raw_model_name.replace('.joblib', '')} predicts that you are {prediction_label} with confidence {confidence:.2f}."
            else:
                prediction = model.predict(Used_X_scaled)
                predicted_user_id = int(prediction[0])
                prediction_label = 'valid' if predicted_user_id == user.id else 'intruder'
                message = f"{raw_model_name.replace('.joblib', '')} predicts that you are {prediction_label}."

            # Append the prediction message for raw models
            prediction_messages.append(message)
            """
        # Save the authentication attempt in the database
        auth_entry = AuthenticationEvaluation(
            user_id=user.id,
            ground_truth='valid' if user_status.lower() == 'valid user' else 'intruder',  # Standardize ground truth as needed
            logistic_regression_prediction=model_predictions['logistic_regression'],
            random_forest_prediction=model_predictions['random_forest'],
            support_vector_machine_prediction=model_predictions['support_vector_machine'],
            gradient_boosting_prediction=model_predictions['gradient_boosting'],
            neural_network_prediction=model_predictions['neural_network'],
            logistic_regression_with_username_prediction=model_predictions['logistic_regression_with_username'],
            random_forest_with_username_prediction=model_predictions['random_forest_with_username'],
            support_vector_machine_with_username_prediction=model_predictions['support_vector_machine_with_username'],
            gradient_boosting_with_username_prediction=model_predictions['gradient_boosting_with_username'],
            neural_network_with_username_prediction=model_predictions['neural_network_with_username']
        )

        db.session.add(auth_entry)
        db.session.commit()

        # Return response to client
        return jsonify({
            'authenticated': True,
            'token': token,
            'predictions': prediction_messages,
            'role': role
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500
    
def extract_features_for_prediction(keystroke_features, processed_data):
    if isinstance(keystroke_features, dict):
        # Extract the raw keystroke features
        press_press_intervals = keystroke_features.get('press_press_intervals', [])
        release_press_intervals = keystroke_features.get('release_press_intervals', [])
        hold_times = keystroke_features.get('hold_times', [])
        total_typing_time = keystroke_features.get('total_typing_time', 0.0)
        typing_speed_cps = keystroke_features.get('typing_speed_cps', 0.0)
        backspace_count = keystroke_features.get('backspace_count', 0)
        error_rate = keystroke_features.get('error_rate', 0.0)

        # Extract processed data (mean and variance values)
        press_press_interval_mean = processed_data.get('press_press_interval_mean', 0)
        press_press_interval_variance = processed_data.get('press_press_interval_variance', 0)
        release_interval_mean = processed_data.get('release_interval_mean', 0)
        release_interval_variance = processed_data.get('release_interval_variance', 0)
        hold_time_mean = processed_data.get('hold_time_mean', 0)
        hold_time_variance = processed_data.get('hold_time_variance', 0)
        press_to_release_ratio_mean = processed_data.get('press_to_release_ratio_mean', 0)

        # Construct the feature vector in the correct order
        feature_vector = (
    press_press_intervals +  # List of press-press intervals
    [press_press_interval_variance] +  # List with the variance
    [typing_speed_cps]  # List with the typing speed
)

        print("Feature vector:", feature_vector)

        # Flatten the list of features (if needed) to avoid nested lists
        flat_features = []
        for item in feature_vector:
            if isinstance(item, list):
                flat_features.extend(item)  # Flatten lists
            else:
                flat_features.append(item)  # Add scalar values directly

        # Convert the flattened feature vector to a DataFrame
        X = pd.DataFrame([flat_features])

        # Check for missing or invalid values
        if X.isnull().values.any() or np.isinf(X.values).any():
            print("Cleaning invalid values in feature set...")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)

        # Load the scaler to ensure correct feature scaling
        scaler_path = r'D:\THESIS\MobileApp\Flask_server\models\raw_scaler.joblib'
        scaler = joblib.load(scaler_path)

        # Get the expected number of features from the scaler (based on its training)
        expected_length = len(scaler.mean_)  # Scaler's mean_ attribute gives the number of features

        # Pad or truncate the feature vector to match the expected length
        current_length = X.shape[1]
        if current_length < expected_length:
            padding = np.zeros((1, expected_length - current_length))  # Create zero padding
            X = np.hstack((X, padding))  # Pad the feature vector
        elif current_length > expected_length:
            X = X.iloc[:, :expected_length]  # Truncate the feature vector

        # Standardize features using the loaded scaler
        X_scaled = pd.DataFrame(scaler.transform(X))

        return X_scaled
    else:
        raise ValueError("Expected keystroke_features to be a dictionary")
def extract_features_for_prediction_username(keystroke_features, username_keystroke_features, username_processed_data, password_processed_data):
    if isinstance(keystroke_features, dict) and isinstance(username_keystroke_features, dict):
        features = []

        features = (

            # Password keystroke feature lists
    keystroke_features.get('press_press_intervals', []) +
    keystroke_features.get('release_press_intervals', []) +
    keystroke_features.get('hold_times', []) +
    
    # Password keystroke statistics (mean and variance)
    [
        password_processed_data.get('press_press_interval_mean', 0),
        password_processed_data.get('press_press_interval_variance', 0),
        password_processed_data.get('hold_time_mean', 0),
        password_processed_data.get('hold_time_variance', 0),
        password_processed_data.get('release_interval_variance', 0)
    ] +
    
    username_keystroke_features.get('press_press_intervals', []) +
    username_keystroke_features.get('release_press_intervals', []) +
    username_keystroke_features.get('hold_times', []) +
    
    
    [
        username_processed_data.get('press_press_interval_mean', 0),
        username_processed_data.get('press_press_interval_variance', 0),
        username_processed_data.get('hold_time_mean', 0),
        username_processed_data.get('hold_time_variance', 0),
        username_processed_data.get('release_interval_variance', 0)
    ] +
    
    
    
    # Scalar features (individual values)
    [
        keystroke_features.get('total_typing_time', 0.0),
        keystroke_features.get('typing_speed_cps', 0.0),  # Typing speed in characters per second
        keystroke_features.get('error_rate', 0.0),  # Assuming error_rate is calculated
        keystroke_features.get('press_to_release_ratio_mean', 0.0)
    ] +
    
    # Username-specific scalar features
    [
        username_keystroke_features.get('total_typing_time', 0.0),
        username_keystroke_features.get('typing_speed_cps', 0.0),  # Typing speed in characters per second
        username_keystroke_features.get('error_rate', 0.0)  # Assuming error_rate is available for username
    ]
)

        # Step 6: Flatten the feature lists and scalar values into a single feature vector
        flat_features = []
        for item in features:
            if isinstance(item, list):
                flat_features.extend(item)  # Flatten lists
            else:
                flat_features.append(item)  # Add scalar values directly

        # Step 7: Convert the flattened feature vector to a DataFrame
        X = pd.DataFrame([flat_features])

        # Step 8: Check for missing or invalid values
        if X.isnull().values.any() or np.isinf(X.values).any():
            print("Cleaning invalid values in feature set...")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(0, inplace=True)

        # Step 9: Load the scaler to ensure correct feature scaling
        scaler_path = r'D:\THESIS\MobileApp\Flask_server\models\Rawusername_scaler.joblib'
        scaler = joblib.load(scaler_path)

        # Step 10: Get the expected number of features from the scaler (based on its training)
        expected_length = len(scaler.mean_)  # Scaler's mean_ attribute gives the number of features

        # Step 11: Pad or truncate the feature vector to match the expected length
        current_length = X.shape[1]
        if current_length < expected_length:
            padding = np.zeros((1, expected_length - current_length))  # Create zero padding
            X = np.hstack((X, padding))  # Pad the feature vector
        elif current_length > expected_length:
            X = X.iloc[:, :expected_length]  # Truncate the feature vector

        # Step 12: Standardize features using the loaded scaler
        X_scaled = pd.DataFrame(scaler.transform(X))

        return X_scaled

    else:
        raise ValueError("Expected keystroke_features and username_keystroke_features to be dictionaries")
