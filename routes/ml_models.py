from flask import Blueprint, request, jsonify
from utils.ml_utils import train_model, train_model_with_username
from utils.ML_training_raw import train_raw_model,train_with_username_data_raw

# This script is used to train the machine learning model by calling both train_model functions

ml_models_bp = Blueprint('ml_models', __name__)

@ml_models_bp.route('/train_model', methods=['POST'])
def train_ml_model():
    try:
        # Retrieve the data from the request
        data = request.get_json()
        max_entries = data.get('max_entries')  # Get max_entries from request 
        use_all_data = data.get('use_all_data', False)  # Default to False if not provided
        

        # If max_entries is provided, ensure that it's a positive number
        if max_entries is not None and max_entries <= 0:
            return jsonify({'error': 'max_entries must be a positive integer'}), 400

        # Check if max_entries is set and either max_entries or use_all_data is valid
        if max_entries is not None or use_all_data:
            # Train both models with the same data limit conditions
            """"
            result3 = train_raw_model(max_entries=max_entries, use_all_data=use_all_data)
            result4= train_with_username_data_raw(max_entries=max_entries, use_all_data=use_all_data)
            """""
            result1 = train_model(max_entries=max_entries, use_all_data=use_all_data)
            result2 = train_model_with_username(max_entries=max_entries, use_all_data=use_all_data)
            
            # Check for errors in the results
            if 'error' in result1:
                return jsonify(result1), 400
            if 'error' in result2:
                return jsonify(result2), 400
            """"
            if 'error' in result3:
                return jsonify(result3), 400
            if 'error' in result4:
                return jsonify(result4), 400
            """
            # Return success message only
            return jsonify({'message': 'Both models trained successfully'})

        else:
            return jsonify({'error': 'Invalid request parameters for training model'}), 400

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 400