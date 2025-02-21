import json

def get_latest_crypto_memes():
    try:
        # Read data from the local coins.json file
        with open("coins.json", "r") as file:
            data = json.load(file)

        # Return a set of tuples (name, symbol) for each coin
        return set((coin["name"], coin["symbol"]) for coin in data)

    except Exception as e:
        print(f"Error fetching crypto meme names and symbols: {e}")
        return set()

# Example usage
memes = get_latest_crypto_memes()
print(memes)
