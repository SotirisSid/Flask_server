import sqlite3
import numpy as np

# Path to your SQLite database
db_path = r'D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db'

# Function to fetch hold_times from keystrokes table, calculate variance, and insert into preprocessed_keystroke_data
def update_hold_time_variance():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to fetch the 'hold_times' column from the 'keystrokes' table
        cursor.execute("SELECT hold_times FROM keystrokes;")
        keystroke_rows = cursor.fetchall()

        # Check if we have the same number of rows in preprocessed_keystroke_data
        cursor.execute("SELECT COUNT(*) FROM preprocessed_keystroke_data;")
        row_count = cursor.fetchone()[0]
        
        if len(keystroke_rows) != row_count:
            print(f"Mismatch in row count: keystrokes ({len(keystroke_rows)}) != preprocessed_keystroke_data ({row_count})")
            return

        # Update the 'hold_times_variance' in preprocessed_keystroke_data for each row
        for i, row in enumerate(keystroke_rows):
            try:
                # Extract and process hold_times (assuming it's stored as a string)
                hold_times_str = row[0]  # Get the hold_times string
                # Convert the string to a list of floats
                hold_times_values = list(map(float, hold_times_str.strip('[]').split(',')))

                # Calculate the variance for the current row
                if len(hold_times_values) > 1:  # Variance needs at least two values
                    variance = np.var(hold_times_values)

                    # Update the corresponding row in preprocessed_keystroke_data table
                    cursor.execute("""
                        UPDATE preprocessed_keystroke_data
                        SET hold_time_variance = ?
                        WHERE rowid = ?;
                    """, (variance, i + 1))  # i + 1 because SQLite rowid starts from 1

                else:
                    print(f"Row {i+1}: Not enough data in hold_times for variance calculation.")
            
            except Exception as e:
                print(f"Error processing row {i+1}: {e}")

        # Commit the transaction to save the changes
        conn.commit()

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
    
    finally:
        # Ensure the connection is closed
        if conn:
            conn.close()





# Path to your SQLite database
db_path = r'D:\THESIS\MobileApp\Flask_server\instance\keystroke_dynamics.db'

# Function to fetch press_press_intervals, calculate variance, and print results
def compute_and_print_press_press_interval_variances():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to fetch the 'press_press_intervals' column from the 'keystrokes' table
        cursor.execute("SELECT press_press_intervals FROM keystrokes;")
        keystroke_rows = cursor.fetchall()

        # Process each row and compute the variance
        for i, row in enumerate(keystroke_rows):
            try:
                # Extract and process press_press_intervals (assuming it's stored as a string)
                press_press_intervals_str = row[0]  # Get the press_press_intervals string
                # Convert the string to a list of floats
                press_press_intervals_values = list(map(float, press_press_intervals_str.strip('[]').split(',')))

                # Calculate the variance for the current row
                if len(press_press_intervals_values) > 1:  # Variance needs at least two values
                    variance = np.var(press_press_intervals_values)
                    print(f"Row {i+1}: Press-Press Interval Variance = {variance:.6f}")
                else:
                    print(f"Row {i+1}: Not enough data in press_press_intervals for variance calculation.")

            except Exception as e:
                print(f"Error processing row {i+1}: {e}")

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
    
    finally:
        # Ensure the connection is closed
        if conn:
            conn.close()

# Call the function to compute and print press-press interval variances
compute_and_print_press_press_interval_variances()