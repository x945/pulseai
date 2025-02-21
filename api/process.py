from flask import Flask, request, jsonify
import spacy
import logging
import json
import re
from textblob import TextBlob  # Import textblob for sentiment analysis

app = Flask(__name__)

# Initialize spaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

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

        # Extract Named Entities using spaCy
        try:
            doc = nlp(truncated_text)
            # Filter useful entities and add to tags set
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PRODUCT", "NORP", "FAC", "LOC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
                    print('NER FOUND:', ent.text)
                    tags.add(ent.text)
            logging.debug("Extracted tags from named entities: %s", tags)
        except Exception as e:
            logging.error("Error during tags extraction: %s", str(e))
            return jsonify({"error": "Failed to extract tags"}), 500

        # Crypto Coins Detection
        try:
            with open("coins.json", "r") as file:
                data = json.load(file)

            crypto_memes = {coin["name"].lower() for coin in data}  # Set for fast lookup

            words = text.split()
            words_cleaned = {re.sub(r"[^\w\s]", "", word.lower()) for word in words}  # Normalize words and remove duplicates

            for word in words_cleaned:
                if word in crypto_memes and all(word.lower() != tag.lower() for tag in tags):
                    tags.add(word.capitalize())  # Capitalize to match expected output
        except Exception as e:
            logging.error("Error during crypto coin detection: %s", str(e))
            return jsonify({"error": "Failed to detect crypto coins"}), 500

        return jsonify({
            "sentiment": {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            },
            "tags": list(tags)  # Convert set to list for JSON serialization
        })

    except Exception as e:
        logging.error("Error processing article: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

