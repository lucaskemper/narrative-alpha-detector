# src/generate_priors.py

import pandas as pd
import time
import re
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("PERPLEXITY_API_KEY")
px = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

def ask_perplexity(question, max_retries=3, min_prob=0.05, max_prob=0.95):
    prompt = (
        "You are a probability estimation assistant.\n\n"
        "Estimate the probability (between 0 and 1) that the following event occurs based on current information. "
        "Also provide a confidence score (between 0 and 1) in your estimate, where 1 means full certainty and 0 means a total guess.\n\n"
        f"Event: {question}\n\n"
        "Respond ONLY in the format:\n"
        "`probability | confidence`\n"
        "For example: `0.27 | 0.75`\n"
        "DO NOT include any explanation, text, or symbols."
    )

    for attempt in range(max_retries):
        try:
            response = px.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            raw = response.choices[0].message.content.strip()
            print(f"🔁 Raw Response: '{raw}'")

            if "|" not in raw:
                print(f"⚠️  Invalid format: missing '|': '{raw}'")
                continue

            prob_str, conf_str = map(str.strip, raw.split("|", 1))
            prob, conf = float(prob_str), float(conf_str)

            if not (0 <= prob <= 1 and 0 <= conf <= 1):
                print(f"⚠️  Invalid numeric range: prob={prob}, conf={conf}")
                continue

            # Clip extreme priors
            prob = min(max(prob, min_prob), max_prob)

            # Confidence-weighted prior (pulls toward 0.5 if unsure)
            adjusted_prob = (1 - conf) * 0.5 + conf * prob

            print(f"✅ Parsed: raw={prob:.3f}, conf={conf:.2f}, adjusted={adjusted_prob:.3f}")
            return adjusted_prob

        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5)

    print(f"❌ Failed to get valid prior after {max_retries} attempts for event:\n{question}")
    return None

def generate_priors(csv_path="results/polymarket_raw.csv", out_path="results/with_priors.csv"):
    df = pd.read_csv(csv_path)
    df["px_prior"] = pd.NA

    # Resume from existing file if present
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        df.update(existing_df)
        print(f"📁 Resuming from existing file: {out_path}")

    for i, row in df.iterrows():
        if not pd.isna(row.get('px_prior')):
            print(f"⏭️  {i+1}/{len(df)}: Skipping (already has prior)")
            continue

        title = row.get("title", "").strip()
        if not title:
            print(f"⚠️  {i+1}: Missing title, skipping")
            continue

        print(f"\n🔮 {i+1}/{len(df)}: {title}")
        prob = ask_perplexity(title)
        if prob is not None:
            df.at[i, "px_prior"] = prob
            df.to_csv(out_path, index=False)
            time.sleep(1.5)  # Rate limit

    print(f"\n✅ Completed and saved to → {out_path}")
    return df

if __name__ == "__main__":
    generate_priors()
