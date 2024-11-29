from flask import Blueprint, request, jsonify
from models import db, User, Keystroke, PreprocessedKeystrokeData
from werkzeug.security import generate_password_hash
import numpy as np
import json
from utils.calculate_features import calculate_keystroke_features  # Import the feature calculation function
from utils.preprocess_keystrokes import process_single_keystroke_data  # Import the preprocessing function
## This script is used to register a new user and store their keystroke data in the database
registration_bp = Blueprint('registration', __name__)

@registration_bp.route('/register_keystrokes', methods=['POST'])
def register_keystrokes():
    try:
        # Get the JSON data from the request
        data = request.json
        username = data['username']
        password = data.get('password')  # Get password from request
        
        # Extract username keystroke data from the request
        username_keypress = request.json.get('key_press_times_username')
        username_keyrelease = request.json.get('key_release_times_username')
        backspace_count_username = int(data.get('backspace_count_username', 0))  # Convert to integer
        
        
        # Validate that username keystroke data is present and has the same length
        if not username_keypress or not username_keyrelease:
            return jsonify({'error': 'Username keystroke data is required.'}), 400
        if len(username_keypress) != len(username_keyrelease):
            return jsonify({'error': 'Username press times and release times must have the same length.'}), 400

        
        
        # Handle times being passed as either list or string, assuming lists from the frontend
        press_times = data.get('key_press_times', [])
        release_times = data.get('key_release_times', [])
        keystroke_intervals = data.get('keystroke_intervals', [])
        
        # Convert comma-separated strings back into lists
        if isinstance(press_times, str):
            press_times = list(map(float, press_times.split(',')))  # Convert strings to floats
            username_keypress = list(map(float, username_keypress.split(',')) )
        if isinstance(release_times, str):
            release_times = list(map(float, release_times.split(',')))  # Convert strings to floats
            username_keyrelease = list(map(float, username_keyrelease.split(',')) )
        if isinstance(keystroke_intervals, str):
            keystroke_intervals = list(map(float, keystroke_intervals.split(',')))  # Convert strings to floats
        
        username_error_rate = (backspace_count_username / (int(len(username_keypress)) + backspace_count_username)) * 100

        # Validate that all key time arrays are present and have the same length
        if len(press_times) != len(release_times):
            return jsonify({'error': 'Press times and release times must have the same length.'}), 400

        if keystroke_intervals and len(keystroke_intervals) != len(press_times) - 1:
            return jsonify({'error': 'Keystroke intervals must be one less than press/release times.'}), 400

        backspace_count = data.get('backspace_count', 0)
        error_rate = data.get('error_rate', 0.0)

        # Check if the username or password is missing
        if not username or not password:
            return jsonify({'error': 'Username and password are required.'}), 400

        # Check if the user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'error': 'Username already exists.'}), 400

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Register a new user with the hashed password
        user = User(username=username, password=hashed_password, role="user")  # Add role="user"
        db.session.add(user)
        db.session.commit()

        # Calculate keystroke features
        keystroke_features = calculate_keystroke_features(press_times, release_times)
        username_keystroke_features = calculate_keystroke_features(username_keypress, username_keyrelease)
        # Create a new keystroke entry with all the collected data
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
    backspace_count_username=backspace_count_username,
    error_rate_username=username_error_rate
)
            # Calculate keystroke features
        keystroke_features = calculate_keystroke_features(press_times, release_times)

        # Extract necessary features from keystroke_features for preprocessing
        press_press_intervals = keystroke_features['press_press_intervals']
        release_press_intervals = keystroke_features['release_press_intervals']
        hold_times = keystroke_features['hold_times']
        total_typing_time = keystroke_features['total_typing_time']
        typing_speed_cps = keystroke_features['typing_speed_cps']
        username_press_press_intervals = username_keystroke_features['press_press_intervals']
        username_release_press_intervals = username_keystroke_features['release_press_intervals']
        username_hold_times = username_keystroke_features['hold_times']
        username_total_typing_time = username_keystroke_features['total_typing_time']
        username_typing_speed_cps = username_keystroke_features['typing_speed_cps']
        # Preprocess keystroke data with correct arguments
        preprocessed_data = process_single_keystroke_data(
            user_id=user.id,
            press_press_intervals=press_press_intervals,
            release_press_intervals=release_press_intervals,
            hold_times=hold_times,
            total_typing_time=total_typing_time,
            typing_speed_cps=typing_speed_cps,
            backspace_count=backspace_count,
            error_rate=error_rate
        )
        username_preprocessed_data = process_single_keystroke_data(
        user_id=user.id,
        press_press_intervals=username_press_press_intervals,
        release_press_intervals=username_release_press_intervals,
        hold_times=username_hold_times,
        total_typing_time=username_total_typing_time,
        typing_speed_cps=username_typing_speed_cps,
        backspace_count=backspace_count_username,
        error_rate=username_error_rate)
        new_preprocessed=PreprocessedKeystrokeData(
        user_id=user.id,
        press_press_interval_mean=preprocessed_data['press_press_interval_mean'],
        release_interval_mean=preprocessed_data['release_interval_mean'],
        hold_time_mean=preprocessed_data['hold_time_mean'],
        press_press_interval_variance=preprocessed_data['press_press_interval_variance'],
        release_interval_variance=preprocessed_data['release_interval_variance'],
        backspace_count=backspace_count,
        error_rate=error_rate,
        # Total typing time
        total_typing_time=keystroke_features['total_typing_time'],
        # Typing speed in characters per second
        typing_speed_cps=keystroke_features['typing_speed_cps'],
        # Username data
        username_press_press_interval_mean=username_preprocessed_data['press_press_interval_mean'],
        username_release_interval_mean=username_preprocessed_data['release_interval_mean'],
        username_hold_time_mean=username_preprocessed_data['hold_time_mean'],
        username_hold_time_variance=username_preprocessed_data['hold_time_variance'],
        username_press_press_interval_variance=username_preprocessed_data['press_press_interval_variance'],
        username_release_interval_variance=username_preprocessed_data['release_interval_variance'],
        username_backspace_count=backspace_count_username,
        username_error_rate=username_error_rate,
        username_total_typing_time=username_total_typing_time,
        username_typing_speed_cps=username_typing_speed_cps
        
    )

        # Save the keystroke data to the database
        db.session.add(keystroke)
        db.session.add(new_preprocessed)
        db.session.commit()

        # If a new user registers, retrain the models
        # request.post('http://localhost:5000/train_model')  # Uncomment if needed

        return jsonify({'message': 'Keystroke data registered successfully'}), 200

    except Exception as e:
        # Improved error handling with more specific feedback
        return jsonify({'error': str(e)}), 400