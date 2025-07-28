# 🧠 Narrative Alpha Detector

LLM-based signal system for detecting retail mispricings in prediction markets

## 🚀 Overview

Narrative Alpha Detector is a pipeline that scans live prediction markets (via Polymarket) and detects potential pricing dislocations by comparing retail odds to AI-generated probability priors.

By querying a large language model (Perplexity Sonar) for calibrated forecasts with built-in confidence, the system ranks trade opportunities where crowd sentiment may diverge from expert-like reasoning.

This is a research-aligned tool for exploring narrative dislocations, sentiment overpricing, and LLM-based alpha surfacing in real time.

## 🧱 Architecture

| Module | Description |
|--------|-------------|
| `src/polymarket_api_scraper.py` | Fetches live prediction markets and odds from Polymarket (Gamma API) |
| `src/generate_priors.py` | Uses Perplexity Sonar to estimate the "true" probability of each event |
| `src/score_mispricing.py` | Compares market vs prior, computes mispricing and expected value |
| `src/analyze_opportunities.py` | Analyzes and filters trading opportunities based on various criteria |
| `src/visualize_opportunities.py` | Creates beautiful charts, dashboards, and reports of trading opportunities |
| `main.py` | Full pipeline runner |
| `results/` | Stores raw, scored, and ranked outputs |

## 🧪 Methodology

**Market Odds**: Polymarket odds are fetched in real time via the Gamma API.

**AI Prior Estimation**: Perplexity's Sonar model is queried with strict prompting and confidence weighting:

```
probability | confidence (e.g., 0.72 | 0.65)
```

Final priors are adjusted via:

```
adjusted = (1 - confidence) * 0.5 + confidence * raw_prob
```

**Scoring**: Mispricing = |Market - Prior|; Expected value = directionally adjusted mispricing

**Filtering**: Expired markets and extreme hallucinations are clipped or skipped


## ⚙️ Setup

```bash
git clone https://github.com/lucaskemper/narrative-alpha-detector.git
cd narrative-alpha-detector
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
PERPLEXITY_API_KEY=your_key_here
```

## 🚀 Usage

```bash
# Full pipeline
python main.py

# Or run individual steps
python src/polymarket_api_scraper.py
python src/generate_priors.py
python src/score_mispricing.py
```

## 💡 Why It Matters

- Prediction markets reflect retail consensus.
- LLMs, when prompted carefully, represent a composite of expert knowledge + internet priors.
- This project surfaces where those diverge — and highlights where narrative-driven mispricings might occur.

## 📌 Notes

- LLMs are not reliable oracles. Priors are clipped and adjusted for realism.
- This is not a trading system, but a research tool for sentiment monitoring and signal experimentation.
- Certain categories (e.g., celebrity gossip, meme ETFs) are prone to hallucinated priors — these are flagged in postprocessing.

## 📈 Future Improvements

- Ensemble multiple LLMs (e.g., GPT-4 + Claude) for more robust priors
- Add outcome tracking + calibration curves
- Deploy real-time dashboard with daily alpha summaries
- Track narrative regime shifts using LLM embeddings
