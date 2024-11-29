from flask import Blueprint, jsonify
from utils.preprocess_keystrokes import preprocess_keystroke_data

preprocess_bp = Blueprint('keystroke', __name__)

@preprocess_bp.route('/preprocess_keystrokes', methods=['POST'])
def preprocess_keystrokes():
    try:
        preprocess_keystroke_data()  # Call the preprocessing function
        return jsonify({"message": "Keystroke data preprocessing completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
