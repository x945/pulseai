import os
import json
import logging
import re
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))

import os
print("NLTK Data Path:", nltk.data.path)
print("Files in NLTK Data Directory:", os.listdir(os.path.join(os.path.dirname(__file__), "..", "nltk_data")))

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def extract_named_entities(text):
    """Extract Named Entities using NLTK."""
    named_entities = set()
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    chunks = ne_chunk(pos_tags)

    for chunk in chunks:
        if isinstance(chunk, Tree):  # Check if chunk is a named entity
            entity = " ".join(c[0] for c in chunk)
            named_entities.add(entity)
    
    return named_entities

@app.route("/process", methods=["POST"])
def process_article():
    logging.debug("Received request: %s", request.get_json())

    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            logging.warning("Text is missing in the request")
            return jsonify({"error": "Text is required"}), 400

        # Truncate text to prevent exceeding model limits
        truncated_text = text[:1024]

        # Initialize tags set
        tags = set()

        # Sentiment Analysis using NLTK's Vader
        try:
            sentiment_scores = sia.polarity_scores(truncated_text)
            sentiment = {
                "polarity": sentiment_scores["compound"],  # Compound score represents overall sentiment
                "positivity": sentiment_scores["pos"],
                "negativity": sentiment_scores["neg"],
                "neutrality": sentiment_scores["neu"]
            }
            logging.debug("Sentiment analysis result: %s", sentiment)
        except Exception as e:
            logging.error("Error during sentiment analysis: %s", str(e))
            return jsonify({"error": "Failed to perform sentiment analysis"}), 500

        # Extract Named Entities using NLTK
        try:
            named_entities = extract_named_entities(truncated_text)
            tags.update(named_entities)
            logging.debug("Extracted tags from named entities: %s", tags)
        except Exception as e:
            logging.error("Error during NER extraction: %s", str(e))
            return jsonify({"error": "Failed to extract named entities"}), 500

        # Crypto Coins Detection
        try:
            coins_file_path = os.path.join(os.path.dirname(__file__), "static", "coins.json")

            with open(coins_file_path, "r") as file:
                data = json.load(file)

            crypto_names = {coin["name"].lower() for coin in data}  # Set for fast lookup

            words_cleaned = {re.sub(r"[^\w\s]", "", word.lower()) for word in text.split()}  # Normalize words

            for word in words_cleaned:
                if word in crypto_names and all(word.lower() != tag.lower() for tag in tags):
                    tags.add(word.capitalize())  # Capitalize for better formatting
        except Exception as e:
            logging.error("Error during crypto coin detection: %s", str(e))
            return jsonify({"error": "Failed to detect crypto coins"}), 500

        return jsonify({
            "sentiment": sentiment,
            "tags": list(tags)  # Convert set to list for JSON serialization
        })

    except Exception as e:
        logging.error("Error processing article: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/", methods=["GET"])
def index():
    return 'PulseAI Server'

# Run Flask app locally
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
