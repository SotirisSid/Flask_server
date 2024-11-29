import os
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import jwt  # Assuming you're using JWT for tokens

#This script is used to logout the user from the system

logout_bp = Blueprint('logout', __name__)

# Token validation function using JWT
def validate_token(token):
    try:
        jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False



@logout_bp.route('/logout', methods=['POST'])

def logout():
    # Get the token from the Authorization header
    token = request.headers.get('Authorization')

    
    if token:
        # You can also check if the token starts with 'Bearer ' if that's part of your implementation
        if token.startswith('Bearer '):
            token = token.split(' ')[1]  # Extract the actual token value
        
        # Validate the token (replace with your actual validation logic)
        if validate_token(token):
            # Proceed with logout logic (e.g., invalidate the token if necessary)
            print(f"Valid token received")
            return jsonify({'message': 'Logged out successfully'}), 200
        else:
            print("Invalid token received during logout attempt.")
            return jsonify({'error': 'Invalid token'}), 401
    else:
        print("Token is missing from the request headers.")
        return jsonify({'error': 'Token is missing'}), 401