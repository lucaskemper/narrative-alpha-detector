# src/score_mispricing.py

import pandas as pd

def score_mispricing(input_path="results/with_priors.csv", output_path="results/scored_markets.csv"):
    df = pd.read_csv(input_path)

    # Sanity check
    df = df[df["prob_yes"] <= 1]
    df = df[df["px_prior"].notnull()]

    # Compute delta and edge
    df["mispricing"] = df["px_prior"] - df["prob_yes"]
    df["abs_mispricing"] = df["mispricing"].abs()
    df["direction"] = df["mispricing"].apply(lambda x: "buy_yes" if x > 0 else "buy_no")

    df = df.sort_values("abs_mispricing", ascending=False)

    df.to_csv(output_path, index=False)
    print(f"✅ Ranked and saved to {output_path}")
    return df

if __name__ == "__main__":
    score_mispricing()
