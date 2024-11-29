import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sklearn.preprocessing import StandardScaler
from scipy import stats
import ast
from models import db, Keystroke  # Ensure you import your Keystroke model


def convert_to_float_list(string_list):
    try:
        return [float(i) for i in ast.literal_eval(string_list)]
    except (ValueError, SyntaxError) as e:
        print(f"Error converting '{string_list}': {e}")
        return []


def convert_to_float_list_sec(input_data):
    try:
        # If input_data is a string representation of a list
        if isinstance(input_data, str):
            # Remove brackets and split by commas
            input_data = input_data.strip('[]').split(',')
        # Convert each element to float
        float_list = [float(item) for item in input_data]
        return float_list
    except Exception as e:
        print(f"Error converting to float list: {e}")
        return []
def preprocess_keystroke_data():
    # Step 1: Create a new SQLAlchemy session using the existing db instance
    Session = sessionmaker(bind=db.engine)
    session = Session()

    try:
        # Step 2: Query the database using SQLAlchemy
        keystroke_data = session.query(
            Keystroke.user_id,
            Keystroke.key_press_times,
            Keystroke.key_release_times,
            Keystroke.backspace_count,
            Keystroke.error_rate
        ).all()

        # Convert SQLAlchemy query result into a pandas DataFrame
        data = pd.DataFrame(keystroke_data, columns=['user_id', 'key_press_times', 'key_release_times', 'backspace_count', 'error_rate'])
        print("Data before preprocessing:")
        

        # Step 3: Convert key press and release times from comma-separated strings to lists of floats


        data['key_press_times'] = data['key_press_times'].apply(convert_to_float_list)
        data['key_release_times'] = data['key_release_times'].apply(convert_to_float_list)

        # Step 4: Calculate metrics from the key press and release times for each session
        data['press_time_mean'] = data['key_press_times'].apply(lambda x: np.mean(x) if x else 0)
        data['release_time_mean'] = data['key_release_times'].apply(lambda x: np.mean(x) if x else 0)
        data['interval_time_mean'] = data.apply(lambda row: np.mean(np.diff(row['key_press_times'])) if len(row['key_press_times']) > 1 else 0, axis=1)
        data['press_time_variance'] = data['key_press_times'].apply(lambda x: np.var(x) if x else 0)
        data['release_time_variance'] = data['key_release_times'].apply(lambda x: np.var(x) if x else 0)
        data['interval_time_variance'] = data.apply(lambda row: np.var(np.diff(row['key_press_times'])) if len(row['key_press_times']) > 1 else 0, axis=1)

        # Step 5: Handle Missing Values
        data['press_time_mean'].fillna(data['press_time_mean'].median(), inplace=True)
        data['release_time_mean'].fillna(data['release_time_mean'].median(), inplace=True)
        data['interval_time_mean'].fillna(data['interval_time_mean'].median(), inplace=True)
        data['backspace_count'].fillna(0, inplace=True)  # Assume no backspace if data is missing
        data['error_rate'].fillna(data['error_rate'].median(), inplace=True)

        # Step 6: Outlier Detection and Removal
        z_scores = np.abs(stats.zscore(data[['press_time_mean', 'release_time_mean', 'interval_time_mean', 'backspace_count', 'error_rate']].fillna(0)))
        data_cleaned = data[(z_scores < 3).all(axis=1)].copy()  # Use .copy() to avoid setting values on a slice
        print("Shape of data after outlier removal:", data_cleaned.shape)

        # Step 7: Feature Scaling and Normalization (Optional)
        if data_cleaned.empty:
            print("data_cleaned is empty. Cannot scale.")
        else:
            scaler = StandardScaler()
            data_cleaned[['press_time_mean', 'release_time_mean', 'interval_time_mean']] = scaler.fit_transform(
                data_cleaned[['press_time_mean', 'release_time_mean', 'interval_time_mean']]
            )

        # Step 8: Feature Extraction - Press to Release Ratio
        data_cleaned['press_to_release_ratio'] = data_cleaned['press_time_mean'] / data_cleaned['release_time_mean']

        # Step 9: Prepare Session Data for Insertion
        session_data = data_cleaned[['user_id', 'press_time_mean', 'release_time_mean', 
                                     'interval_time_mean', 'press_time_variance', 'release_time_variance', 
                                     'interval_time_variance', 'backspace_count', 'error_rate', 'press_to_release_ratio']].copy()

        # Step 10: Save Preprocessed Data into the 'preprocessed_keystroke_data' table in the same database
        session_data.to_sql('preprocessed_keystroke_data', con=db.engine, if_exists='append', index=False)

        print("Preprocessing complete. Session data is saved to the 'preprocessed_keystroke_data' table in the keystroke_dynamics.db database and ready for model training.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
    finally:
        session.close()  # Ensure the session is closed regardless of whether an error occurred


def process_single_keystroke_data(user_id, press_press_intervals, release_press_intervals, hold_times, total_typing_time, typing_speed_cps, backspace_count, error_rate):
    
    

    try:
        # Calculate means and variances from the provided intervals and durations
        press_press_interval_mean = np.mean(
            press_press_intervals) if press_press_intervals else 0

        release_interval_mean = np.mean(
            release_press_intervals) if release_press_intervals else 0
        hold_time_mean = np.mean(hold_times) if hold_times else 0

        # Variances
        press_press_interval_variance = np.var(
            press_press_intervals) if press_press_intervals else 0
        release_interval_variance = np.var(
            release_press_intervals) if release_press_intervals else 0
        hold_time_variance = np.var(hold_times) if hold_times else 0

        # Create a dictionary to hold the processed metrics
        processed_data = {
            'user_id': user_id,
            'press_press_interval_mean': press_press_interval_mean,
            'release_interval_mean': release_interval_mean,
            'hold_time_mean': hold_time_mean,
            'press_press_interval_variance': press_press_interval_variance,
            'release_interval_variance': release_interval_variance,
            'hold_time_variance': hold_time_variance,
            'backspace_count': backspace_count,
            'error_rate': error_rate,
            'total_typing_time': total_typing_time,
            'typing_speed_cps': typing_speed_cps
        }

        return processed_data

    except Exception as e:
        print(f"Error during single keystroke data processing: {e}")
        return None
