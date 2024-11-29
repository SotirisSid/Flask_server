from flask import Blueprint, request, jsonify
from models import db, User, Keystroke, PreprocessedKeystrokeData
from utils.preprocess_keystrokes import process_single_keystroke_data
from werkzeug.security import check_password_hash
import numpy as np
import requests
import json

# Import the feature calculation function
from utils.calculate_features import calculate_keystroke_features

# Create blueprint
train_keystroke_bp = Blueprint('train_keystroke', __name__)


@train_keystroke_bp.route('/train-keystroke', methods=['POST'])
def train_keystroke():
    data = request.get_json()

    # Extract data from the request
    user_name = data.get('userName')
    password = data.get('password')
    key_press_times = data.get('key_press_times')  # List of key press times
    # List of key release times
    key_release_times = data.get('key_release_times')
    backspace_count = data.get('backspace_count')
    error_rate = data.get('error_rate')  # Error rate
    # Extract the data for the username
    username_keypress = data.get('key_press_times_username')
    username_keyrelease = data.get('key_release_times_username')
    backspace_count_username = data.get('backspace_count_username')
    username_error_rate = (backspace_count_username /
                           (int(len(username_keypress)) + backspace_count_username))*100
    print(f'backspace: {backspace_count}')

    # Query the database to retrieve the user's actual stored password hash
    user = User.query.filter_by(username=user_name).first()

    if not user:
        return jsonify({'error': 'Incorrect username'}), 404

    # Verify the raw password with the stored hash
    if not check_password_hash(user.password, password):
        return jsonify({'error': 'Incorrect password'}), 401

    # Find the user id
    user_id = user.id

    # Calculate keystroke features
    keystroke_features = calculate_keystroke_features(
        key_press_times, key_release_times)
    username_keystroke_features = calculate_keystroke_features(
        username_keypress, username_keyrelease)

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
        user_id=user_id,
        press_press_intervals=press_press_intervals,
        release_press_intervals=release_press_intervals,
        hold_times=hold_times,
        total_typing_time=total_typing_time,
        typing_speed_cps=typing_speed_cps,
        backspace_count=backspace_count,
        error_rate=error_rate
    )
    username_preprocessed_data = process_single_keystroke_data(
        user_id=user_id,
        press_press_intervals=username_press_press_intervals,
        release_press_intervals=username_release_press_intervals,
        hold_times=username_hold_times,
        total_typing_time=username_total_typing_time,
        typing_speed_cps=username_typing_speed_cps,
        backspace_count=backspace_count_username,
        error_rate=username_error_rate)

    # Create a new PreprocessedKeystrokeData entry
    new_preprocessed_data = PreprocessedKeystrokeData(
        user_id=user_id,
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

    # Create a new Keystroke record
    new_keystroke = Keystroke(
        user_id=user_id,
        press_press_intervals=json.dumps(
            keystroke_features['press_press_intervals']),  # Added for new structure
        release_press_intervals=json.dumps(
            keystroke_features['release_press_intervals']),  # Added for new structure
        hold_times=json.dumps(
            keystroke_features['hold_times']),  # Added hold times
        # Total typing time
        total_typing_time=keystroke_features['total_typing_time'],
        # Typing speed in characters per second
        typing_speed=keystroke_features['typing_speed_cps'],
        backspace_count=backspace_count,
        error_rate=error_rate,
        # Mean press-to-release ratio
        press_to_release_ratio_mean=keystroke_features['press_to_release_ratio_mean'],
        username_press_press_intervals=json.dumps(
            username_keystroke_features['press_press_intervals']),
        username_release_press_intervals=json.dumps(
            username_keystroke_features['release_press_intervals']),
        username_hold_times=json.dumps(
            username_keystroke_features['hold_times']),
        username_total_typing_time=username_keystroke_features['total_typing_time'],
        username_typing_speed_cps=username_keystroke_features['typing_speed_cps'],
        backspace_count_username=backspace_count_username,
        error_rate_username=username_error_rate
    )

    # Add the keystroke data to the database
    db.session.add(new_keystroke)
    db.session.add(new_preprocessed_data)
    db.session.commit()
    

    return jsonify({'message': 'Keystroke dynamics data added successfully'})
