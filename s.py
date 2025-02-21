from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
import spacy
import logging
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model=model_name)
nlp = spacy.load("en_core_web_md")

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

        # Sentiment Analysis
        try:
            sentiment = sentiment_pipeline(truncated_text)[0]  # Ensuring max input size
            logging.debug("Sentiment analysis result: %s", sentiment)
        except Exception as e:
            logging.error("Error during sentiment analysis: %s", str(e))
            return jsonify({"error": "Failed to perform sentiment analysis"}), 500

        # Extract Named Entities using spaCy
        try:
            doc = nlp(truncated_text)
            # Filter useful entities and add to tags set
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PRODUCT", "NORP", "FAC", "LOC", "EVENT", "WORK_OF_ART", "LAW"]:
                    tags.add(ent.text)
            logging.debug("Extracted tags from named entities: %s", tags)
        except Exception as e:
            logging.error("Error during tags extraction: %s", str(e))
            return jsonify({"error": "Failed to extract tags"}), 500

        # Crypto Coins Detection
        try:
            words = text.split()
            for word in words:
                for coin in crypto_memes:
                    if word.lower() == coin["symbol"] or word.lower() == coin["name"].lower():
                        tags.add(word)
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


@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Get the data from the request
        data = request.get_json()
        news_articles = data.get("newsArticles", "")

        # Check if newsArticles is a string and not empty
        if not news_articles:
            return jsonify({"error": "News Articles must be a non-empty string"}), 400
        
        input_length = len(news_articles.split())
        max_length = max(20, int(input_length * 0.5))

        # Generate summary using BART
        summary_result = summarizer(news_articles[:1024], max_length=max_length, min_length=max(10, max_length // 2), do_sample=False)
        summary_text = summary_result[0]['summary_text']

        # Perform sentiment analysis on the summary
        sentiment = sentiment_pipeline(summary_text)[0]

        # Return the generated summary and sentiment result
        return jsonify({
            "analysis": summary_text,
            "sentiment": sentiment["label"],
            "score": sentiment["score"]
        })

    except Exception as e:
        logging.error("Error generating text: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")  # Allow external connections
