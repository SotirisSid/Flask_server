import requests
import json

# Set the URL of your Flask server
url = 'http://192.168.1.3:5000/register_keystrokes'

# Prepare the data to send
data = {
    "username": "user1",
    "password": "password123",
    "key_press_times": [100, 200, 300],   # Send as list of numbers
    "key_release_times": [150, 250, 350], # Send as list of numbers
    "keystroke_intervals": [50, 50],  # Send as list of numbers
    "backspace_count": 2,
    "error_rate": 0.05
}
print(data)

# Send the POST request
try:
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        print('Success:', response.json())
    else:
        print('Failed:', response.status_code, response.json())

except Exception as e:
    print('An error occurred:', str(e))
