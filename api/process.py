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

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Define a set of stopwords (meaningless tags to remove)
stopwords_set = {
    "latest", "now", "week", "month", "today", "breaking", "update", "hour", "news", "recent", "current",
    "daily", "weekly", "monthly", "yearly", "day", "night", "morning", "evening", "midnight", "afternoon",
    "minute", "second", "soon", "moment", "while", "before", "after", "yesterday", "tomorrow", "tonight",
    "headline", "trending", "live", "watch", "hot", "flash", "instant", "special", "exclusive", "featured",
    "reported", "report", "coverage", "announcement", "story", "storyline", "press", "article", "blog",
    "read", "written", "editorial", "post", "shared", "link", "source", "details", "info", "information",
    "about", "regarding", "summary", "analysis", "discussion", "opinion", "reaction", "review", "explained",
    "revealed", "find", "found", "according", "statement", "quoted", "declared", "mentioned", "noted",
    "breaking news", "hot topic", "must see", "viral", "thread", "threaded", "debate", "argument", "trending now",
    "alert", "emergency", "warning", "exclusive update", "this just in", "up next", "coming soon", "stay tuned",
    "insights", "reveals", "insider", "leak", "rumor", "speculation", "guess", "assumption", "potential", "expected",
    "possibly", "alleged", "reportedly", "unconfirmed", "maybe", "likely", "prediction", "forecast", "outlook", "column",
    "bitfinex", "alpha", "new", "crypto", "bitcoin news", "house", "west", "million", "high", "surge", "top", "no",
    "video", "coin", "us", "drops", "serb", "trailer", "black", "activist", "first", "cash", "match", "best", "why",
    "news digest", "how", "home", "health", "dawn", "boost", "super", "safe", "tested", "sir", "guide", "future",
    "make", "answer", "will", "which", "when", "value", "secret", "react", "price", "pay", "network", "into", "human",
    "hidden", "help", "everyone", "dark", "card", "channel", "key", "film", "way", "visible", "unknown", "trend",
    "tips", "the", "takes", "system", "show", "secrets", "season", "same", "save", "role", "robust", "rise", "return",
    "reality", "has", "fresh", "for", "could", "would", "response"
}

tickers = {
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "BRK.A", "TSLA", "AVGO", "LLY",
    "WMT", "JPM", "V", "MA", "XOM", "ORCL", "COST", "UNH", "NFLX", "AMD",
    "PG", "ABBV", "CVX", "MRK", "PEP", "KO", "ADBE", "TMO", "ASML", "PFE",
    "NKE", "INTC", "CSCO", "CRM", "DHR", "MCD", "ACN", "LIN", "TXN", "AMAT",
    "NOW", "NFLX", "UPS", "HON", "QCOM", "SBUX", "GS", "IBM", "C", "INTU",
    "MDT", "LOW", "BLK", "CAT", "ELV", "CHTR", "NEE", "SPGI", "T", "ISRG",
    "PLD", "DE", "LRCX", "BA", "MO", "BKNG", "VRTX", "REGN", "GE", "ADI",
    "MS", "CI", "GILD", "AXP", "BSX", "FISV", "TJX", "PGR", "PYPL", "SYK",
    "HUM", "FIS", "SNPS", "CME", "D", "ADP", "GM", "TGT", "CSX", "DUK",
    "ZTS", "SO", "BDX", "WM", "CL", "MMC", "ICE", "EW", "EQIX", "EMR"
}

companies = {
    "Apple",
    "NVIDIA",
    "Microsoft",
    "Amazon",
    "Alphabet",
    "Saudi Aramco",
    "Meta",
    "Tesla",
    "Berkshire Hathaway",
    "Broadcom",
    "TSMC",
    "Eli Lilly",
    "Walmart",
    "JPMorgan",
    "Visa",
    "Tencent",
    "Mastercard",
    "Exxon",
    "Oracle",
    "Costco",
    "UnitedHealth",
    "Netflix",
    "Procter & Gamble",
    "Novo Nordisk",
    "Johnson & Johnson",
    "Home Depot",
    "LVMH",
    "AbbVie",
    "Bank of America",
    "SAP",
    "ICBC",
    "Coca-Cola",
    "T-Mobile",
    "Hermès",
    "Salesforce",
    "ASML",
    "Toyota",
    "Chevron",
    "Samsung",
    "Roche",
    "Kweichow Moutai",
    "Wells Fargo",
    "Cisco",
    "Agricultural Bank of China",
    "Pfizer",
    "L'Oréal",
    "Abbott",
    "AMD",
    "Adobe",
    "Novartis",
    "Reliance",
    "McDonald's",
    "Blackstone",
    "American Express",
    "HSBC",
    "Intuit",
    "Qualcomm",
    "Texas Instruments",
    "Verizon",
    "PepsiCo",
    "Caterpillar",
    "Raytheon",
    "Booking Holdings",
    "S&P Global",
    "Intuitive Surgical",
    "Morgan Stanley",
    "Tata Consultancy Services",
    "Linde",
    "Thermo Fisher",
}

football = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion", "Burnley",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Newcastle United", "Nottingham Forest", "Sheffield United",
    "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
    "Aberdeen", "Celtic", "Dundee", "Dundee United", "Heart of Midlothian", "Hibernian",
    "Kilmarnock", "Livingston", "Motherwell", "Rangers", "Ross County", "St. Johnstone", "St. Mirren",
    "Aberystwyth Town", "Airbus UK Broughton", "Bala Town", "Barry Town United", "Caernarfon Town",
    "Cardiff Metropolitan University", "Connah's Quay Nomads", "Flint Town United",
    "Haverfordwest County", "Newtown", "Penybont", "The New Saints",
    "Ballymena United", "Carrick Rangers", "Cliftonville", "Coleraine", "Crusaders",
    "Dungannon Swifts", "Glenavon", "Glentoran", "Larne", "Linfield", "Newry City", "Portadown",
    "Real Madrid", "Barcelona", "PSG", "Bayern", "Juve",
    "Atleti", "Milan", "Inter", "Dortmund", "Napoli",
    "RB Leipzig", "Sevilla", "Roma", "Lazio", "Bayer Leverkusen",
    "Lyon", "Ajax", "Benfica", "Shakhtar", "Porto",
    "Sporting", "Villarreal", "Atalanta", "Galatasaray", "Fenerbahçe",
    "Marseille", "Zenit", "Salzburg", "Dynamo Kyiv", "Brugge",
    "Flamengo", "River", "Boca", "São Paulo", "Palmeiras",
    "Santos", "Grêmio", "Monterrey", "Tigres", "Al Ahly",
    "Zamalek", "Sundowns", "Al-Hilal", "Jeonbuk", "Kashima", 
    "Ulsan", "Club América", "Independiente", "Peñarol", "Nacional"
}



def extract_named_entities(text):
    """Extract Named Entities using NLTK and ensure they are capitalized."""
    named_entities = set()
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    chunks = ne_chunk(pos_tags)

    for chunk in chunks:
        if isinstance(chunk, Tree):  # Check if chunk is a named entity
            entity = " ".join(c[0] for c in chunk)
            named_entities.add(entity.title())  # Ensures proper capitalization
    
    return named_entities

@app.route("/process", methods=["POST"])
def process_article():
    logging.debug(f"Received request: {request.get_json()}")
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        category = data.get("category")

        if not text:
            logging.warning("Text is missing in the request")
            return jsonify({"error": "Text is required"}), 400

        truncated_text = text[:1024]  # Truncate text to prevent exceeding model limits

        tags = set()

        # Sentiment Analysis using NLTK's Vader
        try:
            sentiment_scores = sia.polarity_scores(truncated_text)
            sentiment = {
                "polarity": sentiment_scores["compound"],  
                "positivity": sentiment_scores["pos"],
                "negativity": sentiment_scores["neg"],
                "neutrality": sentiment_scores["neu"]
            }
            logging.debug(f"Sentiment analysis result: {sentiment}")
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            return jsonify({"error": "Failed to perform sentiment analysis"}), 500

        # Extract Named Entities using NLTK
        try:
            named_entities = extract_named_entities(truncated_text)
            tags.update(named_entities)
            logging.debug(f"Extracted tags from named entities: {tags}")
        except Exception as e:
            logging.error(f"Error during NER extraction: {e}")
            return jsonify({"error": "Failed to extract named entities"}), 500
        
        # Perform Tickers Search only if category is 'markets'
        if category and category.lower() == "markets":
            try:
                words_cleaned = {re.sub(r"[^\w\.\-]", "", word) for word in text.split()}  # Preserve "." and "-"
                
                for word in words_cleaned:
                    if word in (tickers | companies) and word not in tags:
                        tags.add(word)  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during tickers search: {e}")
                return jsonify({"error": "Failed to search tickers"}), 500

        # Perform Crypto Coins Detection only if category is 'cryptocurrency'
        if category and category.lower() == "cryptocurrency":
            try:
                coins_file_path = os.path.join(os.path.dirname(__file__), "static", "coins.json")

                with open(coins_file_path, "r") as file:
                    coin_data = json.load(file)

                crypto_names = {coin["name"].lower() for coin in coin_data}  # Set for fast lookup
                words_cleaned = {re.sub(r"[^\w\s]", "", word.lower()) for word in text.split()}  # Normalize words

                for word in words_cleaned:
                    if word in crypto_names and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.capitalize())  # Capitalize for better formatting
            except Exception as e:
                logging.error(f"Error during crypto coin detection: {e}")
                return jsonify({"error": "Failed to detect crypto coins"}), 500

        if category and category.lower() == "sports":
            try:
                words_cleaned = {re.sub(r"[^\w\.\-]", "", word) for word in text.split()}  # Preserve "." and "-"
                
                for word in words_cleaned:
                    if word in football and word not in tags:
                        tags.add(word)  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during tickers search: {e}")
                return jsonify({"error": "Failed to search tickers"}), 500

        # **Filter Out Stopwords from Tags**
        tags = {tag for tag in tags if tag.lower() not in stopwords_set}

        return jsonify({
            "sentiment": sentiment,
            "tags": list(tags)  # Convert set to list for JSON serialization
        })

    except Exception as e:
        logging.error(f"Error processing article: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/", methods=["GET"])
def index():
    return 'PulseAI Server'
