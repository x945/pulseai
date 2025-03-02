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
    "reality", "has", "fresh", "for", "could", "would", "response", 'use', 'reviewed', 'ethereum news', 'asset', 'big', 
    'major', 'reach', 'zone', 'trust', 'among first', 'worst month', 'that', 'strong', 'unprecedented', 'price movements',
    'sex scenes', 'woman', 'times asks', 'sex trafficking charges', 'says', 'rush', 'part', 'offset', 'live stream', 'women',
    'earth', 'meet', 'odds'
}

agencies = {
    'acf', 'acl', 'ada', 'adf', 'afrh', 'ahrq', 'aid', 'alat', 'ambc', 'ams', 'aphis', 'arc', 'ars', 
    'atf', 'bea', 'bia', 'bja', 'bjs', 'blm', 'bls', 'bop', 'bpa', 'bts', 'cbo', 'cbp', 'cdc', 'cftc', 
    'cgr', 'cigie', 'cio', 'cisa', 'cit', 'cja', 'cms', 'cncs', 'cpsc', 'crs', 'csb', 'csosa', 'cvc', 
    'dhs', 'dia', 'dla', 'doc', 'dod', 'doe', 'doi', 'doj', 'dol', 'dos', 'dot', 'dpc', 'dtra', 'eac', 
    'ebsa', 'eda', 'ed', 'eeoc', 'eere', 'eia', 'epa', 'erda', 'esa', 'faa', 'fas', 'fbi', 'fbop', 
    'fcc', 'fda', 'fdic', 'fec', 'fema', 'ferc', 'fhfa', 'fhlb', 'firb', 'fletc', 'fmcsa', 'fmc', 
    'fmcs', 'fmshrc', 'fncs', 'fns', 'fpc', 'fra', 'frb', 'fsa', 'fsis', 'fss', 'ftc', 'fws', 'gao', 
    'gsa', 'hhs', 'hhs-oig', 'his', 'hmd', 'hpc', 'hrsa', 'hud', 'iaf', 'ice', 'ida', 'ifc', 'imf', 
    'ins', 'interpol', 'irs', 'ita', 'itc', 'jag', 'jcs', 'jpl', 'lac', 'lra', 'mcc', 'mda', 'mms', 
    'msha', 'nasa', 'nara', 'nass', 'ncua', 'nea', 'neh', 'nga', 'nhtsa', 'nibin', 'nic', 'nifa', 
    'nih', 'nist', 'nlrb'
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
    "arsenal", "aston villa", "bournemouth", "brentford", "brighton & hove albion", "burnley",
    "chelsea", "crystal palace", "everton", "fulham", "liverpool", "luton town", "manchester city",
    "manchester united", "newcastle united", "nottingham forest", "sheffield united",
    "tottenham hotspur", "west ham united", "wolverhampton wanderers",
    "aberdeen", "celtic", "dundee", "dundee united", "heart of midlothian", "hibernian",
    "kilmarnock", "livingston", "motherwell", "rangers", "ross county", "st. johnstone", "st. mirren",
    "aberystwyth town", "airbus uk broughton", "bala town", "barry town united", "caernarfon town",
    "cardiff metropolitan university", "connah's quay nomads", "flint town united",
    "haverfordwest county", "newtown", "penybont", "the new saints",
    "ballymena united", "carrick rangers", "cliftonville", "coleraine", "crusaders",
    "dungannon swifts", "glenavon", "glentoran", "larne", "linfield", "newry city", "portadown",
    "real madrid", "barcelona", "psg", "bayern", "juve",
    "atleti", "milan", "inter", "dortmund", "napoli",
    "rb leipzig", "sevilla", "roma", "lazio", "bayer leverkusen",
    "lyon", "ajax", "benfica", "shakhtar", "porto",
    "sporting", "villarreal", "atalanta", "galatasaray", "fenerbahçe",
    "marseille", "zenit", "salzburg", "dynamo kyiv", "brugge",
    "flamengo", "river", "boca", "são paulo", "palmeiras",
    "santos", "grêmio", "monterrey", "tigres", "al ahly",
    "zamalek", "sundowns", "al-hilal", "jeonbuk", "kashima", 
    "ulsan", "club américa", "independiente", "peñarol", "nacional"
}

leagues = {
    "NFL", "NBA", "MLB", "NHL", "MLS",
    "WNBA", "NWSL", "CFL", "XFL", "USFL",
    "MLR", "PLL", "NLL", "USL", "USL1",
    "USL2", "MLSNP", "AHL", "ECHL", "G League",
    "ATP", "WTA", "PGA", "LIV", "INDYCAR",
    "NASCAR"
}

crypto = {
    'btc', 'eth', 'xrp', 'bnb', 'sol', 'doge', 'ada', 'steth', 'shib', 'hbar',
    'om', 'hype', 'dot', 'bch', 'bgb', 'uni', 'plsx', 'xmr', 'wbt', 'near',
    'pepe', 'aave', 'tao', 'ondo', 'apt', 'icp', 'tkx', 'etc', 'mnt',
    'gt', 'okb', 's', 'bopb', 'vet', 'fet', 'jup', 'op', 'inj', 'kcs', 'ldo',
    'reth', 'qnt', 'stx', 'jto', 'virtual', 'bera', 'ron', 'cake', 'spx',
    'hnt', 'crv', 'pls', 'axs'
}

us_teams = {
    'afa', 'akr', 'ala', 'app', 'ariz', 'asu', 'ark', 'arst', 'army', 'aub',
    'ball', 'bay', 'bc', 'bgsu', 'bsu', 'buff', 'byu', 'cal', 'ccu', 'char',
    'cin', 'clem', 'cmu', 'colo', 'csu', 'duke', 'ecu', 'emu', 'fau', 'fiu',
    'fla', 'fsu', 'fres', 'gaso', 'gast', 'gt', 'haw', 'hou', 'idho', 'ill',
    'ind', 'iowa', 'isu', 'ksu', 'ku', 'ken', 'lib', 'lou', 'lt', 'lsu', 'mem',
    'mia', 'miao', 'mich', 'minn', 'miss', 'miz', 'msu', 'mtu', 'navy', 'ncst',
    'nd', 'neb', 'nev', 'nmsu', 'ntx', 'nw', 'odu', 'ohio', 'okla', 'okst',
    'orst', 'ou', 'pitt', 'psu', 'pur', 'rice', 'rut', 'sala', 'sdsu', 'smu',
    'sou', 'stan', 'syr', 'ta&m', 'tcu', 'tem', 'tenn', 'tex', 'tol', 'troy',
    'tuln', 'tuls', 'uab', 'ucf', 'ucla', 'uga', 'uk', 'um', 'umass', 'unc',
    'unlv', 'unt', 'usc', 'usf', 'usm', 'utep', 'utsa', 'uva', 'van', 'vt',
    'wash', 'wazu', 'wcu', 'wis', 'wmu', 'wvu', 'wyo'
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
            named_entities.add(entity)  # Ensures proper capitalization
    
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

        except Exception as e:
            logging.error(f"Error during tickers search: {e}")
            return jsonify({"error": "Failed to search tickers"}), 500
        
        # Perform Tickers Search only if category is 'markets'
        if category and category.lower() == "markets":
            try:
                words_cleaned = {re.sub(r"[^\w\.\-]", "", word) for word in text.split()}  # Preserve "." and "-"
                
                for word in words_cleaned:
                    if word.title() in companies and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Keep tickers as they are
                    if word.upper() in tickers and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())  # Keep tickers as they are
                    if word.lower() in crypto and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())  # Keep tickers as they are
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
                words_cleaned = {re.sub(r"[^\w\s]", "", word) for word in text.split()}  # Normalize words

                for word in words_cleaned:
                    if word in crypto_names and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Capitalize for better formatting

                    if word.lower() in crypto and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during crypto coin detection: {e}")
                return jsonify({"error": "Failed to detect crypto coins"}), 500

        if category and category.lower() == "sports":
            try:
                words_cleaned = {re.sub(r"[^\w\.\-]", "", word.lower()) for word in text.split()}  # Preserve "." and "-"
                
                for word in words_cleaned:
                    if word.lower() in football and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Keep tickers as they are
                for word in words_cleaned:
                    if word.lower() in us_teams and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Keep tickers as they are
                for word in words_cleaned:
                    if word.lower() in leagues and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during tickers search: {e}")
                return jsonify({"error": "Failed to search tickers"}), 500

        # **Filter Out Stopwords from Tags**
        tags = {tag for tag in tags if tag.lower() not in stopwords_set}

        # call upper on agencies tags
        final_tags = {tag.upper() if tag.lower() in agencies else tag for tag in tags}

        return jsonify({
            "sentiment": sentiment,
            "tags": list(final_tags)  # Convert set to list for JSON serialization
        })

    except Exception as e:
        logging.error(f"Error processing article: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/", methods=["GET"])
def index():
    return 'PulseAI Server'
