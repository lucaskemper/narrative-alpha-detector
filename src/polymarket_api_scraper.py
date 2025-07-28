# src/polymarket_gamma_scraper.py

import requests
import pandas as pd
import json

GAMMA_ENDPOINT = "https://gamma-api.polymarket.com/markets"

def fetch_polymarket_markets(limit=100):
    """
    Fetch active Polymarket markets from the Gamma API.
    
    Args:
        limit (int): Maximum number of markets to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing market data with columns:
            - title: Market question
            - category: Market category
            - url: Market URL
            - prob_yes: Probability of YES outcome
            - prob_no: Probability of NO outcome
            - end_date: Market end date
            - active: Whether market is active
            - closed: Whether market is closed
    """
    print("🔍 Fetching markets from Gamma API...")
    
    # Use query parameters to get active, non-closed markets
    params = {
        'limit': limit,
        'active': 'true',
        'closed': 'false'  # Only get non-closed markets
    }
    
    response = requests.get(GAMMA_ENDPOINT, params=params)
    data = response.json()
    
    # Handle different possible response structures
    if isinstance(data, list):
        markets_raw = data[:limit]
    elif isinstance(data, dict) and "markets" in data:
        markets_raw = data["markets"][:limit]
    elif isinstance(data, dict) and "data" in data:
        markets_raw = data["data"][:limit]
    else:
        print(f"Unexpected response structure: {type(data)}")
        return pd.DataFrame()
    
    rows = []

    for m in markets_raw:
        if not isinstance(m, dict):
            continue
            
        if "outcomes" not in m or len(m["outcomes"]) < 2:
            continue

        # Extract YES/NO probabilities if available
        yes_price = None
        no_price = None
        
        # Check if we have outcomePrices (newer API format)
        if "outcomePrices" in m:
            outcome_prices = m["outcomePrices"]
            if isinstance(outcome_prices, dict):
                yes_price = outcome_prices.get("YES")
                no_price = outcome_prices.get("NO")
            elif isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                # If outcomePrices is a list, assume first two elements are YES/NO
                yes_price = outcome_prices[0] if len(outcome_prices) > 0 else None
                no_price = outcome_prices[1] if len(outcome_prices) > 1 else None
        else:
            # Try to parse outcomes as strings (older format)
            for outcome in m["outcomes"]:
                if not isinstance(outcome, dict):
                    continue
                    
                if outcome.get("name", "").lower() == "yes":
                    yes_price = outcome.get("price")
                elif outcome.get("name", "").lower() == "no":
                    no_price = outcome.get("price")

        # If we don't have valid prices from outcomePrices, try alternative sources
        if yes_price is None or no_price is None or (str(yes_price) == "0" and str(no_price) == "0"):
            # Try to use lastTradePrice or bestBid/bestAsk
            last_trade = m.get("lastTradePrice")
            best_bid = m.get("bestBid")
            best_ask = m.get("bestAsk")
            
            if last_trade and last_trade != "0":
                yes_price = last_trade
                no_price = 1 - float(last_trade) if isinstance(last_trade, (int, float)) else None
            elif best_bid and best_bid != "0":
                yes_price = best_bid
                no_price = 1 - float(best_bid) if isinstance(best_bid, (int, float)) else None
            elif best_ask and best_ask != "0":
                yes_price = best_ask
                no_price = 1 - float(best_ask) if isinstance(best_ask, (int, float)) else None

        # Only include markets with valid prices
        if yes_price is not None and no_price is not None:
            try:
                yes_float = float(yes_price)
                no_float = float(no_price)
                if yes_float > 0 or no_float > 0:  # At least one price should be non-zero
                    rows.append({
                        "title": m.get("question", "Unknown"),
                        "category": m.get("category", "unknown"),
                        "url": f"https://polymarket.com/market/{m.get('slug', 'unknown')}",
                        "prob_yes": yes_price,
                        "prob_no": no_price,
                        "end_date": m.get("endDate", "n/a"),
                        "active": m.get("active", "unknown"),
                        "closed": m.get("closed", "unknown")
                    })
            except (ValueError, TypeError):
                continue

    df = pd.DataFrame(rows)
    df.to_csv("results/polymarket_raw.csv", index=False)
    print(f"✅ Saved {len(df)} markets to results/polymarket_raw.csv")
    return df

if __name__ == "__main__":
    df = fetch_polymarket_markets()
    print(df.head())
