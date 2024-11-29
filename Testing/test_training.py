import requests

def test_train_model():
    # Set the URL of your Flask application
    url = "http://localhost:5000/train_model"  # Update with the correct URL if needed

    try:
        # Send a POST request to the /train_model route
        response = requests.post(url)

        # Check the response status code
        if response.status_code == 200:
            print("Test passed: Model trained successfully")
            print("Response:", response.json())  # Print the response from the server
        else:
            print(f"Test failed: Received HTTP status code {response.status_code}")
            print("Response:", response.json())  # Print the response for further details
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_train_model()