import requests
import json

# Define the URL for the authentication endpoint
url = "http://localhost:5000/authenticate"

# Define the payload with the username, password, and keystroke data
payload = {
    "username": "user1",
    "password": "password123",
    "key_press_times": [100, 120, 110],
    "key_release_times": [200, 220, 210],
    "backspace_count": 1,
    "error_rate": 0.05
}

# Send a POST request to the authentication endpoint
response = requests.post(url, json=payload)

# Print the response from the server
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())