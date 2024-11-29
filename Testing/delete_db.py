from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
# Replace 'your_database.db' with the path to your SQLite database file
DATABASE_URI = 'sqlite:///D:/THESIS/MobileApp/Flask_server/instance/keystroke_dynamics.db'

# Create a database engine
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()
print("Current Working Directory:", os.getcwd())
print("Database URI:", DATABASE_URI)

try:
    # List of tables to delete entries from
    tables = ['keystrokes', 'preprocessed_keystroke_data', 'sqlite_sequence', 'users']
    
    for table in tables:
        # Execute the DELETE command with text()
        session.execute(text(f'DELETE FROM {table}'))
    
    session.commit()  # Commit the changes
    print("All entries have been deleted from the specified tables.")
    
except Exception as e:
    print(f"An error occurred: {e}")
    session.rollback()  # Roll back the session in case of an error
finally:
    session.close()  # Close the session