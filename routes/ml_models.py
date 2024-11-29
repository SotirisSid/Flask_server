from flask import Blueprint, request, jsonify
from utils.ml_utils import train_model,train_model_with_username

##this script is used to train the machine learning model by calling the train_model function

ml_models_bp = Blueprint('ml_models', __name__)

@ml_models_bp.route('/train_model', methods=['POST'])
def train_ml_model():
    try:
        train_model()
        train_model_with_username()

        return jsonify({'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
