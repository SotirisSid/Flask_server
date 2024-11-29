import requests

# URL of the Flask server
BASE_URL = 'http://localhost:5000'  # Adjust if your server is running on a different host or port

def test_preprocess_keystrokes():
    try:
        response = requests.post(f'{BASE_URL}/preprocess_keystrokes')
        response_data = response.json()
        
        if response.status_code == 200:
            print("Success:", response_data['message'])
        else:
            print("Error:", response_data.get('error', 'Unknown error occurred'))
    except requests.exceptions.RequestException as e:
        print("Failed to connect to the server:", str(e))

if __name__ == '__main__':
    test_preprocess_keystrokes()
