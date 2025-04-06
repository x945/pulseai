import os
import json
import logging
import re
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Initialize NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

financial_markets_keywords = {
    "stock", "stocks", "market", "markets", "nasdaq", "dow", "s&p", "sp500", "ftse", "bond", "bonds",
    "treasury", "yield", "interest rate", "inflation", "recession", "gdp", "economy", "economic",
    "federal reserve", "fed", "rate hike", "earnings", "ipo", "index", "indices", "commodities",
    "oil", "gold", "futures", "etf", "mutual fund", "hedge fund", "portfolio", "dividend", "capital",
    "valuation", "debt", "equity", "buyback", "quarterly", "bear", "bull", "volatility", "investors",
    "retail", "institutional", "trading", "financials", "analyst", "forecast", "macro", "micro",
    "ai stocks", "tech rally", "bond selloff", "soft landing", "jobs report", "labor market",
    "inflation data", "rate cut", "fed meeting"
}

cryptocurrency_keywords = {
    "crypto", "blockchain", "web3", "coin", "token", "defi", "nft", "bitcoin", "ethereum", "altcoin",
    "wallet", "exchange", "binance", "coinbase", "crypto market", "mining", "staking", "airdrop",
    "smart contract", "gas fees", "dex", "cex", "metamask", "ledger", "cold storage", "tokenomics",
    "halving", "crypto trading", "rug pull", "crypto crash", "bull run", "bear market",
    "bitcoin etf", "eth upgrade", "layer 2", "ordinals", "zk rollups", "defi lending"
}

us_news_keywords = {
    "biden", "trump", "white house", "congress", "senate", "house of representatives", "democrats",
    "republicans", "election", "midterms", "supreme court", "lawsuit", "governor", "state", "capitol",
    "senator", "representative", "shooting", "storm", "flood", "wildfire", "police", "fbi", "crime",
    "immigration", "border", "irs", "debt ceiling", "government shutdown", "military", "pentagon",
    "campaign", "infrastructure", "healthcare", "social security", "medicare", "veterans", "national",
    "2024 elections", "primary debates", "trump trial", "border crisis", "school shootings"
}

technology_keywords = {
    "ai", "artificial intelligence", "machine learning", "deep learning", "data science", "cloud",
    "saas", "software", "hardware", "semiconductor", "chip", "quantum", "robot", "automation",
    "startups", "tech", "gadgets", "app", "apps", "app store", "play store", "android", "ios",
    "smartphone", "laptop", "device", "internet", "cybersecurity", "hack", "breach", "meta", "google",
    "microsoft", "amazon", "elon musk", "tesla", "openai", "chatgpt", "neuralink", "apple", "spacex",
    "generative ai", "sora", "groq", "ai race", "ai regulation", "data leak",
}

world_affairs_keywords = {
    "un", "united nations", "war", "conflict", "peace", "diplomacy", "treaty", "summit", "nato",
    "g7", "g20", "embassy", "foreign minister", "ambassador", "geopolitics", "russia", "china",
    "ukraine", "iran", "north korea", "israel", "palestine", "gaza", "refugee", "human rights",
    "migration", "protest", "coup", "election", "diplomatic", "sanction", "regime", "international",
    "red sea attacks", "taiwan tensions", "hezbollah", "icj ruling", "gaza war"
}

entertainment_keywords = {
    # General
    "celebrity", "celebrities", "gossip", "drama", "entertainment", "hollywood", "red carpet", "scandal",

    # Movies & TV
    "movie", "film", "box office", "blockbuster", "series", "tv", "tv show", "netflix", "hbo", "prime video",
    "disney+", "apple tv", "hulu", "trailer", "premiere", "sequel", "reboot", "remake", "director",
    "actor", "actress", "cast", "scene", "script", "screenplay", "filmmaker", "cinema", "production",
    "studio", "release date", "ratings",

    # Music
    "music", "song", "single", "album", "track", "billboard", "grammys", "mtv", "spotify", "itunes",
    "tour", "concert", "festival", "musician", "artist", "rapper", "singer", "dj", "band", "pop", "rap",
    "hip hop", "rock", "edm", "country", "lyrics", "remix",

    # Awards
    "oscars", "academy awards", "golden globes", "emmys", "grammy", "bafta", "sundance", "cannes",

    # Celebrities & Trends
    "kardashian", "taylor swift", "beyoncé", "drake", "rihanna", "kanye", "bad bunny", "selena gomez",
    "harry styles", "zendaya", "timothée chalamet", "tom holland", "margot robbie", "barbie movie",
    "social media", "viral", "influencer", "reality show", "met gala", "fashion", "award show", "red carpet"
}

sports_keywords = {
    "match", "goal", "tournament", "final", "score", "league", "athlete", "coach", "referee",
    "injury", "stadium", "champion", "world cup", "olympics", "nba", "nfl", "mlb", "nhl", "ufc",
    "mma", "boxing", "fifa", "epl", "la liga", "serie a", "champions league", "draft", "transfer",
    "trade", "playoff", "season", "game", "win", "loss", "draw",
    "march madness", "super bowl", "nfl draft", "f1", "cricket world cup", "injury report"
}

science_keywords = {
    "research", "study", "scientists", "experiment", "data", "climate", "physics", "chemistry",
    "biology", "genetics", "lab", "discovery", "innovation", "medical", "vaccine", "pandemic",
    "covid", "health", "medicine", "quantum", "space", "nasa", "james webb", "earthquake",
    "volcano", "weather", "planet", "galaxy", "neuroscience", "disease", "biology", "biotech",
    "climate tipping point", "ozone hole", "covid variant", "ai", "asteroid approach"
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
    "aapl", "nvda", "msft", "amzn", "googl", "meta", "brk.a", "tsla", "avgo", "lly",
    "wmt", "jpm", "v", "ma", "xom", "orcl", "cost", "unh", "nflx", "amd",
    "pg", "abbv", "cvx", "mrk", "pep", "ko", "adbe", "tmo", "asml", "pfe",
    "nke", "intc", "csco", "crm", "dhr", "mcd", "acn", "lin", "txn", "amat",
    "now", "ups", "hon", "qcom", "sbux", "gs", "ibm", "c", "intu",
    "mdt", "low", "blk", "cat", "elv", "chtr", "nee", "spgi", "t", "isrg",
    "pld", "de", "lrcx", "ba", "mo", "bkng", "vrtx", "regn", "ge", "adi",
    "ms", "ci", "gild", "axp", "bsx", "fisv", "tjx", "pgr", "pypl", "syk",
    "hum", "fis", "snps", "cme", "d", "adp", "gm", "tgt", "csx", "duk",
    "zts", "so", "bdx", "wm", "cl", "mmc", "ice", "ew", "eqix", "emr"
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

leagues = {
    "nfl", "nba", "mlb", "nhl", "mls",
    "wnba", "nwsl", "cfl", "xfl", "usfl",
    "mlr", "pll", "nll", "usl", "usl1",
    "usl2", "mlsnp", "ahl", "echl", "g league",
    "atp", "wta", "pga", "liv", "indycar",
    "nascar"
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

category_keywords = {
    "markets": financial_markets_keywords,
    "cryptocurrency": cryptocurrency_keywords,
    "us": us_news_keywords,
    "technology": technology_keywords,
    "world": world_affairs_keywords,
    "entertainment": entertainment_keywords,
    "sports": sports_keywords,
    "science": science_keywords
}

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

        # Keyword matching based on category
        if category in category_keywords:
            try:
                for keyword in category_keywords[category]:
                    if keyword in text.lower():
                        tags.add(keyword.title())

                words_cleaned = {re.sub(r"[^\w\s]", "", word) for word in text.split()}
                for word in words_cleaned:
                    if word.lower() in agencies and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())
            except Exception as e:
                    logging.error(f"Error during keyword search: {e}")
                    return jsonify({"error": "Failed to search keywords"}), 500

        # Perform Tickers Search only if category is 'markets'
        if category and category.lower() == "markets":
            try:
                # Match full company names in text
                for company in companies:
                    if company in text and all(company.lower() != tag.lower() for tag in tags):
                        tags.add(company)

                words_cleaned = {re.sub(r"[^\w\.\-]", "", word) for word in text.split()}  # Preserve "." and "-"
                for word in words_cleaned:
                    if word.lower() in tickers and all(word.lower() != tag.lower() for tag in tags):
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
                    if word.lower() in crypto_names and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.title())  # Capitalize for better formatting
                    if word.lower() in crypto and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during crypto coin detection: {e}")
                return jsonify({"error": "Failed to detect crypto coins"}), 500

        if category and category.lower() == "sports":
            try:
                # Match full football club names in text
                for club in football:
                    if club in text and all(club.lower() != tag.lower() for tag in tags):
                        tags.add(club)

                words_cleaned = {re.sub(r"[^\w\.\-]", "", word) for word in text.split()}  # Preserve "." and "-"
                for word in words_cleaned:
                    if word.lower() in us_teams and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word)  # Keep tickers as they are
                for word in words_cleaned:
                    if word.lower() in leagues and all(word.lower() != tag.lower() for tag in tags):
                        tags.add(word.upper())  # Keep tickers as they are
            except Exception as e:
                logging.error(f"Error during tickers search: {e}")
                return jsonify({"error": "Failed to search tickers"}), 500

        custom_formatting = {
            # Financial & Markets
            "s&p": "S&P",
            "sp500": "S&P 500",
            "ftse": "FTSE",
            "etf": "ETF",
            "ipo": "IPO",
            "gdp": "GDP",

            # Cryptocurrency
            "nft": "NFT",
            "defi": "DeFi",
            "dex": "DEX",
            "cex": "CEX",
            "metamask": "MetaMask",
            "defi lending": "DeFi Lending",

            # Technology
            "ai": "AI",
            "chatgpt": "ChatGPT",
            "openai": "OpenAI",
            "generative ai": "Generative AI",
            "ai race": "AI Race",
            "ai regulation": "AI Regulation",
            "neuralink": "Neuralink",

            # World Affairs
            "un": "UN",
            "icj ruling": "ICJ Ruling",

            # Entertainment
            "tv": "TV",
            "tv show": "TV Show",
            "hbo": "HBO",
            "dj": "DJ",
            "edm": "EDM",
            
            # Science
            "covid": "COVID",
            "covid variant": "COVID Variant",
            "ai": "AI",
        }

        final_tags = set()
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in custom_formatting:
                final_tags.add(custom_formatting[tag_lower])
            else:
                final_tags.add(tag)


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
