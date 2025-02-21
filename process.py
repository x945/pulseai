import requests
import json

# Define the URL of your Flask server
url = "http://127.0.0.1:5000/process"  # Replace with your actual URL if different

# Sample text to send to the /process endpoint
sample_text = """
Stocks making the biggest moves midday: Hanesbrands, Reddit, Nvidia, AppLovin and more
"""

# Prepare the payload (data) as a JSON object
payload = {
    "text": sample_text
}

# Headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send a POST request to the /process endpoint
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Check if the response was successful
if response.status_code == 200:
    print("Response from server:")
    print(response.json())  # Print the JSON response from the server
else:
    print(f"Failed to process the article. Status code: {response.status_code}")
    print(response.text)
