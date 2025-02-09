from flask import Flask, request, jsonify
from transformers import pipeline
from keybert import KeyBERT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize models
sentiment_pipeline = pipeline("sentiment-analysis")
kw_model = KeyBERT()

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Parse JSON input
        data = request.get_json()
        news_articles = data.get("newsArticles", "")
        video_content = data.get("videoContent", "")
        tag = data.get("tag", "")

        if not news_articles or not video_content:
            return jsonify({"error": "Both newsArticles and videoContent are required"}), 400

        if not tag:
            return jsonify({"error": "Tag is required"}), 400

        # Combine the text from both sources
        combined_text = f"Provide a very short glance of the current situation regarding '{tag}' based on the following news articles and video content. Use a maximum of two lines to describe the key themes.\nNews Articles: {news_articles}\nVideo Content: {video_content}"

        # Tokenize input text
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)

        # Generate output text from the model
        with torch.no_grad():
            output = model.generate(inputs["input_ids"], max_length=100)  # Short output
          
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Perform sentiment analysis
        sentiment = sentiment_pipeline(generated_text)[0]  # {"label": "POSITIVE", "score": 0.99}
        
        return jsonify({
            "generated_text": generated_text,
            "sentiment": sentiment["label"],
            "score": sentiment["score"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def process_article():
    data = request.get_json()
    text = data.get("text")

    # Perform sentiment analysis
    sentiment = sentiment_pipeline(text)[0]  # {"label": "POSITIVE", "score": 0.99}

    # Extract keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')

    # Return sentiment and keywords
    return jsonify({
        "sentiment": sentiment,
        "tags": keywords
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')  # Allow external connections
