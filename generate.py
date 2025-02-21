import requests
import json

# Define the URL for the Flask app's /generate endpoint
url = "http://127.0.0.1:5000/generate"  # Change this to your actual Flask server URL

# Input text (news articles) to be sent to the /generate endpoint
news_articles = ['Citi Lifts CEO Jane Fraser’s Pay by a Third to $34.5 Million',
 'Ackman Dangles $900 Million in New Bid to Revamp Howard Hughes',
 'NYC’s 590 Madison Goes Up for Sale at Roughly $1.1 Billion',
 'China’s Tech Rally Turns Pony AI Founder Into a Billionaire']

# Prepare the payload with the input data
payload = {
    "newsArticles": news_articles
}

# Set headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send the POST request to the Flask app
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    response_data = response.json()
    print("Generated Summary: ", response_data.get("analysis"))
    print("Sentiment: ", response_data.get("sentiment"))
    print("Sentiment Score: ", response_data.get("score"))
else:
    print(f"Error: {response.status_code}, {response.text}")
