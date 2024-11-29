from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import db  # Adjust if your db instance is elsewhere
from models import User  # Assuming your User model is in app.models
import os
from utils.evaluate_metrics import evaluate_model_for_all_users,plot_metrics

# Define your database URI here
DATABASE_URI = 'sqlite:///D:/THESIS/MobileApp/Flask_server/instance/keystroke_dynamics.db'

# Blueprint for the admin routes
admin_bp = Blueprint('admin', __name__)

# Route to reset the database
@admin_bp.route('/admin/reset-database', methods=['POST'])
@jwt_required()  # Ensure the user is authenticated
def reset_database():
    print("Authorization Header:", request.headers.get('Authorization'))
    # Get the current user identity (username) from the token
    current_user_id = get_jwt_identity()

    # Verify the user is an admin
    admin_user = User.query.filter_by(username=current_user_id, role='admin').first()
    if not admin_user:
        return jsonify({"error": "Unauthorized"}), 403

    # Connect to the database and perform reset
    engine = create_engine(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Specify tables to reset, excluding the admin information in 'users'
        tables_to_clear = ['keystrokes', 'preprocessed_keystroke_data', 'sqlite_sequence']

        # Clear data from the specified tables
        for table in tables_to_clear:
            session.execute(text(f'DELETE FROM {table}'))

        # Optionally, reset the user data excluding admin entries
        session.execute(text("DELETE FROM users WHERE role != 'admin'"))

        # Commit the transaction
        session.commit()
        return jsonify({"message": "Database reset successfully"}), 200

    except Exception as e:
        session.rollback()
        return jsonify({"error": f"An error occurred: {e}"}), 500

    finally:
        session.close()

@admin_bp.route('/admin/evaluate-metrics', methods=['GET'])
@jwt_required()  # Ensure the user is authenticated
def evaluate_metrics():
    # Get the current user identity (username) from the token
    current_user_id = get_jwt_identity()

    # Verify if the user is an admin
    admin_user = User.query.filter_by(username=current_user_id, role='admin').first()
    if not admin_user:
        return jsonify({"error": "Unauthorized"}), 403
    # Call the function to get the metrics
    try:
        metrics = evaluate_model_for_all_users()
        plot_metrics(metrics)

        return jsonify({"evaluation_metrics": metrics}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
