from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Enum
db = SQLAlchemy()  # For keystroke_dynamics.db


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(80), nullable=False)

class AuthenticationEvaluation(db.Model):
    __tablename__ = 'authentication_evaluation'
    
    attempt_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String, nullable=False)
    
    # Ground truth column using Enum to validate 'valid' or 'intruder'
    ground_truth = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'), nullable=False)
    
    # Model prediction columns with validation for 'valid' or 'intruder' using Enum
    logistic_regression_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    random_forest_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    support_vector_machine_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    gradient_boosting_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    neural_network_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    
    logistic_regression_with_username_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    random_forest_with_username_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    support_vector_machine_with_username_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    gradient_boosting_with_username_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))
    neural_network_with_username_prediction = db.Column(Enum('valid', 'intruder', name='valid_intruder_enum'))

class Keystroke(db.Model):
    __tablename__ = 'keystrokes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    press_press_intervals = db.Column(db.String, nullable=True)
    release_press_intervals = db.Column(db.String, nullable=True)
    hold_times = db.Column(db.String, nullable=True)
    total_typing_time = db.Column(db.Float, nullable=True)
    typing_speed = db.Column(db.Float, nullable=True)
    backspace_count = db.Column(db.Integer, nullable=True)
    error_rate = db.Column(db.Float, nullable=True)
    press_to_release_ratio_mean = db.Column(db.Float)
    username_press_press_intervals = db.Column(db.String, nullable=True)
    username_release_press_intervals = db.Column(db.String, nullable=True)
    username_hold_times = db.Column(db.String, nullable=True)
    username_total_typing_time = db.Column(db.Float, nullable=True)
    username_typing_speed_cps = db.Column(db.Float, nullable=True)
    backspace_count_username = db.Column(db.Integer, nullable=True)
    error_rate_username = db.Column(db.Float, nullable=True)


class PreprocessedKeystrokeData(db.Model):
    __tablename__ = 'preprocessed_keystroke_data'

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, primary_key=True)
    
    # Columns for password typing data
    press_press_interval_mean = db.Column(db.Float, nullable=False)
    release_interval_mean = db.Column(db.Float, nullable=False)
    hold_time_mean = db.Column(db.Float, nullable=False, default=0.0)
    press_press_interval_variance = db.Column(db.Float, nullable=False)
    release_interval_variance = db.Column(db.Float, nullable=False)
    hold_time_variance = db.Column(db.Float, nullable=False, default=0.0)
    backspace_count = db.Column(db.Integer, nullable=False)
    error_rate = db.Column(db.Float, nullable=False)
    total_typing_time = db.Column(db.Float, nullable=False, default=0.0)
    typing_speed_cps = db.Column(db.Float, nullable=False, default=0.0)
    
    # Columns for username typing data
    username_press_press_interval_mean = db.Column(db.Float, nullable=False, default=0.0)
    username_release_interval_mean = db.Column(db.Float, nullable=False, default=0.0)
    username_hold_time_mean = db.Column(db.Float, nullable=False, default=0.0)
    username_press_press_interval_variance = db.Column(db.Float, nullable=False, default=0.0)
    username_release_interval_variance = db.Column(db.Float, nullable=False, default=0.0)
    username_hold_time_variance = db.Column(db.Float, nullable=False, default=0.0)
    username_backspace_count = db.Column(db.Integer, nullable=False, default=0)
    username_error_rate = db.Column(db.Float, nullable=False, default=0.0)
    username_total_typing_time = db.Column(db.Float, nullable=False, default=0.0)
    username_typing_speed_cps = db.Column(db.Float, nullable=False, default=0.0)