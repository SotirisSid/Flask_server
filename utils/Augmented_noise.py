import sqlite3
import pandas as pd
import numpy as np

# Configuration for noise levels
NOISE_CONFIG = {
    'press_press_intervals': 0.01,
    'press_release_durations': 0.01,
    'release_press_intervals': 0.01,
    'hold_times': 0.005,
    'total_typing_time': 0.02,
    'typing_speed_cps': 0.005,
}

def augment_user_data(user_data, noise_config, num_samples=40):
    """
    Augments data for a single user by adding noise to numeric columns.

    Parameters:
        user_data (pd.DataFrame): Original data for a user.
        noise_config (dict): Noise levels for each feature.
        num_samples (int): Number of synthetic samples to generate.

    Returns:
        pd.DataFrame: Augmented data for the user.
    """
    numeric_columns = user_data.select_dtypes(include=[np.number]).columns
    augmented_data = []
    for _ in range(num_samples):
        noisy_sample = user_data.copy()
        for feature, noise_level in noise_config.items():
            if feature in numeric_columns:
                noisy_sample[feature] += np.random.normal(0, noise_level, size=len(user_data))
        augmented_data.append(noisy_sample)
    return pd.concat(augmented_data, ignore_index=True)

def augment_dataset(dataset, noise_config, num_samples=40):
    """
    Augments the entire dataset by adding synthetic samples for each user.

    Parameters:
        dataset (pd.DataFrame): Original dataset with a user_id column.
        noise_config (dict): Noise levels for each feature.
        num_samples (int): Number of synthetic samples to generate per user.

    Returns:
        pd.DataFrame: Augmented dataset.
    """
    augmented_dataset = []
    for user_id, user_data in dataset.groupby('user_id'):
        print(f"Augmenting data for user_id: {user_id}")
        augmented_data = augment_user_data(user_data, noise_config, num_samples)
        augmented_dataset.append(augmented_data)
    return pd.concat(augmented_dataset, ignore_index=True)

# Database paths
DB_PATH = r"D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db"
SOURCE_TABLE_NAME = "keystrokes"
NEW_TABLE_NAME = "combined_keystrokes"

if __name__ == "__main__":
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    
    # Load data from the source table
    query = f"SELECT * FROM {SOURCE_TABLE_NAME}"
    dataset = pd.read_sql_query(query, conn)

    # Augment dataset
    augmented_dataset = augment_dataset(dataset, NOISE_CONFIG, num_samples=10)

    # Combine original and augmented data
    combined_dataset = pd.concat([dataset, augmented_dataset], ignore_index=True)

    # Save combined data to a new table in the existing database
    combined_dataset.to_sql(NEW_TABLE_NAME, conn, if_exists='replace', index=False)
    print(f"Combined dataset (original + augmented) saved to table '{NEW_TABLE_NAME}' in database '{DB_PATH}'")

    # Close database connection
    conn.close()
