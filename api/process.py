from flask import Flask, request, jsonify
import logging
import json
import re
from textblob import TextBlob  # Import textblob for sentiment analysis

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route("/process", methods=["POST"])
def process_article():
    logging.debug("Received request: %s", request.get_json())

    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            logging.warning("Text is missing in the request")
            return jsonify({"error": "Text is required"}), 400

        # Truncate text to ensure it fits within the model's token limit
        truncated_text = text[:1024]

        # Initialize tags set
        tags = set()

        # Sentiment Analysis using TextBlob
        try:
            blob = TextBlob(truncated_text)
            sentiment = blob.sentiment
            logging.debug("Sentiment analysis result: %s", sentiment)
        except Exception as e:
            logging.error("Error during sentiment analysis: %s", str(e))
            return jsonify({"error": "Failed to perform sentiment analysis"}), 500


        return jsonify({
            "sentiment": {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            }
        })

    except Exception as e:
        logging.error("Error processing article: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

