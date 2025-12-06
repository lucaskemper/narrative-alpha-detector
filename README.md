# Narrative Alpha Detector

> LLM-based signal system for detecting retail mispricings in prediction markets.  

Important: The specific edge this system originally exploited has since disappeared as the prediction-market microstructure evolved. This repository now documents the framework as a research artifact; my current work uses a newer methodology built on different signals and regime structures.

---

## ⚠️ Note  
This repository contains the **public-facing documentation** of the Narrative Alpha Detector.  
The **full production codebase is private** because the system is actively deployed in a live trading environment.  
If you're interested in collaborating, discussing the methodology, or exploring research angles, **please reach out**.

## Overview

Narrative Alpha Detector is a pipeline that scans live prediction markets (via Polymarket) and detects potential pricing dislocations by comparing retail odds to AI-generated probability priors.

By querying large language models (Perplexity Sonar, Grok, GPT & Claude) for calibrated forecasts with built-in confidence, the system ranks trade opportunities where crowd sentiment may diverge from expert-like reasoning.

**It also supports cross-venue arbitrage detection** by comparing odds from Polymarket, Kalshi, PredictIt, and Manifold for identical or near-identical events, highlighting pricing divergences for research and analysis.

---

## Key Capabilities

**Statistical Signal Gating**  
Market odds and priors are converted to Beta belief parameters; log-likelihood ratios (LLR) gate opportunities with tagged failure reasons before any trade sizing.

**Adaptive Ensemble Priors**  
Blend Perplexity, GPT, Grok, Claude, and heuristic forecasters with category-aware adaptive weights, stability-driven confidence decay, and adversarial market skipping.

**Regime & Volatility-Aware Sizing**  
Regime-specific mispricing volatility feeds fractional Kelly sizing with confidence scaling; drawdown and velocity clamps tune leverage and can stop trading via the kill switch.

**Liquidity-Aware Execution**  
Non-linear slippage (power-law impact + spread floors/caps) feeds effective prices, trade logs, and portfolio attribution for realistic PnL.

**Resilient Data Pipeline & Arbitrage**  
Fault-tolerant Polymarket ingestion, cross-venue odds from Kalshi/PredictIt/Manifold for divergence detection, and automatic model disablement on repeated provider failures.

**Operational Health & Latency**  
End-to-end timers for LLM and exchange calls emit JSONL latency logs; health monitor plus Streamlit heatmap highlight slow providers, slippage breaches, and trigger the kill switch.

**Shadow Trading & CI**  
Shadow mode logs intended orders without transmitting them; automated validation, Dockerized services, and CI keep deployments safe.

**Opportunity Intelligence & Reporting**  
Regime-aware tagging, NLP-driven sector labels, calibration/stability drift alerts, Streamlit dashboards, and full audit trails for every run.

---

## Documentation

**Signal Filtering Config**  
`config/signal_filters.json` centralizes evaluation and opportunity-filter thresholds (LLR, mispricing, confidence, liquidity, freshness, regimes) and persists evaluation reasons into scored outputs.

**Risk Limits & Sizing Config**  
`config/risk_limits.json` defines bankroll/category caps plus sizing controls (`fractional_kelly`, `target_mispricing_vol`, `max_kelly`), drawdown/velocity limits, leverage scaling, and execution/slippage parameters.

**Health & Stability Config**  
`config/health_thresholds.json` sets latency, slippage, drawdown, and anomaly limits for the health monitor and kill switch; stability decay settings (e.g., `tau`, `max_drift_before_disable`) control confidence decay for priors.

---

## Architecture

| Module | Description |
|--------|-------------|
| `src/polymarket_api_scraper.py` | Fetches live prediction markets and odds from Polymarket (Gamma API) |
| `src/score_mispricing.py` | Converts market/prior to Beta params, computes mispricing, edges, log-likelihood ratios, and persists scoring fields |
| `src/signal_evaluator.py` | Applies LLR/mispricing/confidence/liquidity gates, tags evaluation reasons, and annotates pass/fail status |
| `src/log_outcomes.py` | Pulls resolved markets, versioned snapshots, and joins outcomes with predictions |
| `src/generate_priors.py` | Multi-model prior generation with adaptive ensemble weighting, stability-driven confidence decay, and adversarial skipping |
| `src/simulate_portfolio.py` | Backtests automated bet placement with regime mispricing volatility, fractional Kelly sizing, drawdown clamps, and slippage modeling |
| `src/risk_manager.py` | Validates proposed trades against bankroll/category/regime constraints and drawdown/velocity de-risking |
| `src/execute_order.py` | Stateful order agent with transport abstraction, risk checks, and lifecycle logging |
| `src/analyze_opportunities.py` | Analyzes and filters trading opportunities with tagging/meta features and regime mispricing volatility rollups |
| `src/visualize_opportunities.py` | Creates beautiful charts, dashboards, and reports of trading opportunities |
| `src/cross_venue_arb.py` | Fetches odds from Kalshi and PredictIt, compares with Polymarket for arbitrage opportunities, and scores divergences |
| `src/utils/paths.py` | Centralized project paths and directory scaffolding helpers |
| `src/utils/loggers.py` | JSONL logging utilities for prompts, pipeline events, latency, and health data |
| `src/utils/llm_models.py` | Provider-specific LLM adapters (Perplexity, OpenAI, Anthropic, Grok, local heuristic) |
| `src/utils/ensembles.py` | Adaptive ensemble weighting and history persistence |
| `src/utils/regimes.py` | Regime classification heuristics, history tracking, and volatility calculations |
| `src/utils/stability.py` | Distribution drift and calibration monitoring helpers |
| `src/utils/tagging.py` | Keyword + optional LLM tagging helpers (sector, liquidity, horizon) |
| `src/utils/adversarial.py` | Heuristics to flag fragile/adversarial markets and reduce LLM weight or skip them |
| `src/dashboard.py` | Streamlit interactive dashboard (top mispricings, regimes, calibration, PnL, latency heatmap, health) |
| `src/stability_monitor.py` | CLI to compare priors vs baseline, aggregate drift scores, and log alerts |
| `src/health_monitor.py` | Evaluates latency/slippage/anomaly thresholds, drawdown signals, and manages the kill switch |
| `src/aggregate_latency.py` | Aggregates JSONL latency logs into provider/component summaries for the dashboard |
| `src/audit_trail.py` | Consolidates predictions/outcomes/simulations into timestamped JSONL |
| `src/generate_audit_report.py` | Builds HTML/PDF audit reports summarising the latest run |
| `scripts/counterfactual_eval.py` | Counterfactual tester comparing current ensemble to historical weights on current markets |
| `src/trade_reconciler.py` | Final gate for proposed trades combining risk limits, execution filters, and health status |
| `main.py` | Full pipeline runner |
| `results/` | Structured outputs (`raw/`, `priors/`, `scored/`, `figures/`, `reports/`, `logs/`, `prompts/`, etc.) |

---

## Methodology

### Market Odds
Polymarket odds are fetched in real time via the Gamma API.

### AI Prior Estimation
LLMs are queried with strict prompting and confidence weighting:

```
probability | confidence (e.g., 0.72 | 0.65)
```

Final priors are adjusted via:

```
adjusted = (1 - confidence) * 0.5 + confidence * raw_prob
```

Predictions fan out asynchronously across enabled providers with per-model timeouts; any provider that times out or raises an API error is dropped from the run and logged for auditability.

### Statistical Signal Gating
Market prices and ensemble priors are mapped to Beta parameters (`alpha_mkt`, `beta_mkt`, `alpha_prior`, `beta_prior`) to capture belief strength. Log-likelihood ratios (`ll_prior`, `ll_mkt`, `llr`) quantify how much better the prior fits the expected outcome than the market and are saved in `latest_scored_markets.csv` plus timestamped scored outputs.

### Signal Evaluation
`src/signal_evaluator.py` reads `config/signal_filters.json` and labels each scored signal as `passed`/`failed`, tagging `evaluation_reason` for thresholds such as LLR, absolute mispricing, confidence, liquidity, freshness, and market status before downstream filtering.

### Risk Management & Execution

- `src/risk_manager.py` loads `config/risk_limits.json` and enforces portfolio, category, regime, and per-trade caps; leverage multipliers react to drawdown depth and drawdown velocity.
- `src/simulate_portfolio.py` computes regime-specific mispricing volatility, applies confidence-scaled fractional Kelly sizing with caps, adjusts for non-linear slippage, and records approval metadata for every simulated trade.
- `src/trade_reconciler.py` acts as the final pre-submit gate combining proposed trades, risk limits, and kill-switch/health state.
- `src/execute_order.py` wraps live submission with a state machine (`SENT → ACKNOWLEDGED → PARTIAL/FULL_FILL`, `CANCEL_SENT → CANCELLED`, `ERROR`) and gracefully handles API rejections without crashing.

### Operational Monitoring & Kill Switch

- `main.py` logs per-stage latencies and persists end-to-end runtime metrics under `results/health/`.
- Network/LLM wrappers log latency JSONL lines via `src/utils/loggers.py`; `src/aggregate_latency.py` summarizes into `results/health/latency_summary.csv` for the dashboard heatmap.
- `src/health_monitor.py` compares latency, slippage, drawdowns, drawdown velocity, and LLM anomalies to `config/health_thresholds.json`, raising or clearing the kill switch (`results/health/kill_switch.json`) and writing mode (`normal`, `de_risk`, `kill`).
- The Streamlit dashboard (`src/dashboard.py`) surfaces health status, latency heatmap, active breaches, and kill switch state in real time.

## Advanced Mispricing Scoring Engine

The scoring engine is no longer a simple “prior minus market price”.  
It is a full multi-factor alpha model designed to surface real, tradeable edges.

### What the engine computes
Each market receives a composite score combining:

**1. Beta belief strength & log-likelihood ratio**  
Market and prior beliefs are stored as `alpha_*`/`beta_*`; `ll_prior`, `ll_mkt`, and `llr` quantify statistical separation and are logged alongside mispricing.

**2. Confidence-weighted probability mispricing**  
The raw prior–market gap is scaled by ensemble confidence, suppressing noisy priors and amplifying stable ones.

**3. Logit-space dislocation**  
Markets near 0 or 1 behave nonlinearly.  
Logit mispricing highlights structural dislocations that linear deltas completely miss.

**4. Kelly-based edge estimation**  
For each market, the theoretical Kelly fraction is computed for both YES and NO sides.  
The dominant side’s edge becomes part of the composite score.

**5. Liquidity-adjusted weighting**  
Illiquid markets get penalized.  
Highly liquid markets receive a boost based on a capped liquidity factor.

**6. Time-to-expiry decay**  
Events approaching resolution get a natural alpha boost.  
Far-out markets decay toward 0.5 importance to reduce noise.

**7. Regime mispricing volatility**  
Rolling volatility by `regime` and `category` (`mispricing_vol_regime`) captures regime stability for sizing and dashboarding.

**8. Edge bucket segmentation**  
Each opportunity is labeled into one of four buckets for downstream filtering and dashboards:

- tiny  
- scalp  
- solid  
- fat  

### Composite Score Formula (Conceptual)

```text
score = |mispricing| * confidence * logit_boost * liquidity_factor * kelly_edge_factor * time_decay
```
Scores are clipped, versioned, and pushed into:

- `latest_scored_markets.csv`  
- timestamped run outputs  
- full audit logs  

### Filtering
Expired markets, extreme hallucinations, and signals failing LLR/confidence/liquidity/mispricing gates are clipped or tagged out; `evaluation_reason` is persisted for traceability.

### Regime Scoring & Structure

- Each market is scored into `trending`, `news_shock`, `meme`, or `low_attention` regimes using NLP heuristics, volume, and price velocity.
- Regime scores are appended to scored opportunities, logged per run (`results/regimes/`), and rolled up to track transitions where alpha comes/goes.
- Tags incorporate `regime:<label>` so analysts can filter reports, dashboards, and simulations by structural state.

### Stability Monitoring

- Distributional drift (KL, KS, mean/std shift) for priors is compared to the baseline snapshot in `results/stability/baseline.json`; drift scores aggregate KL and Brier deltas per model/category.
- `src/generate_priors.py` applies exponential confidence decay using `tau` and drops models that exceed `max_drift_before_disable`, logging adjustments to ensemble history.
- Reports land in `results/stability/*_stability.json` and counterfactual tests in `results/stability/counterfactual_eval_<date>.json` for audit-ready summaries and daily checks.

### Auditability & Reporting

- `src/audit_trail.py` appends snapshot metadata for priors, scored signals, outcomes, and portfolio trades into `results/logs/audit_trail.jsonl`.
- `src/generate_audit_report.py` renders HTML (and optional PDF when WeasyPrint is installed) summaries combining regime stats, calibration, stability alerts, and portfolio attribution.
- Each run emits a per-trade trace (`results/audit/{run_id}_trade_trace.jsonl`) linking raw snapshots → priors → evaluation → risk/execute events, and the audit report is refreshed automatically.
- Streamlit dashboard (`src/dashboard.py`) provides interactive insights across mispricings, regimes, calibration curves, and simulated PnL.

### Ensemble Modeling

- Multiple models (Perplexity/GPT/Claude/local heuristic) generate priors; each prediction is logged with per-model confidence.
- Historical outcome performance (Brier/log-loss) is tracked by `src/log_outcomes.py` and stored in `results/resolved/model_performance*.csv`.
- `src/generate_priors.py` uses those metrics plus drift scores to adapt ensemble weights per category via inverse-Brier weighting with exponential decay, recording usage in `results/priors/ensemble_history.jsonl`.

### Cross-Venue Arbitrage Detection

- Fetches odds from Kalshi, PredictIt, and Manifold using public APIs (no authentication required).
- Matches events via fuzzy + manual mapping, then compares with Polymarket data to identify divergences (default >5% probability or ~0.3 logit difference).
- Scores and logs opportunities in `results/scored/{run_id}_arb_opps.csv` and `latest_arb_opps.csv` for analysis, without executing trades.
- Especially valuable in 2025 given Manifold's high volume on conditional and long-tail events; routinely surfaces 15–40% dislocations missed by pairwise comparisons.
- Useful for studying market fragmentation, adversarial pricing dynamics, or as input for MEV simulations.

### Versioning & Auditability

- Every Polymarket pull is captured as `results/raw/YYYYMMDD_hhmmss_markets.json` plus a `latest_markets.csv`.
- Prior runs persist to `results/priors/` (timestamped + latest) and legacy compatibility files remain in `results/`.
- JSONL prompt logs land in `results/prompts/`, while pipeline components emit structured event logs to `results/logs/`.
- CLI invocations set a run ID (`NAD_RUN_ID`) that stitches all artifacts together.
- Latency JSONL logs, summaries, and kill-switch state live under `results/health/` for health monitor ingestion and dashboard heatmaps.
- Regime tags and confidence scores are archived in `results/regimes/` for transition analysis.

---

Create a `.env` file with your API keys:

```env
PERPLEXITY_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GROK_API_KEY=your_key_here
```

If a key is missing the model simply drops out of the ensemble (and is noted in the pipeline logs).

The first run will automatically scaffold the `results/` hierarchy (raw snapshots, priors, scored data, figures, reports, logs, prompts, etc.) and a `notebooks/` folder for analysis notebooks.

---

## Usage

### Full Pipeline

```bash
python main.py
```

### Individual Steps

```bash
# Fetch markets
python src/polymarket_api_scraper.py

# Generate priors
python src/generate_priors.py

# Score mispricings
python src/score_mispricing.py

# Apply quality gates
python src/signal_evaluator.py

# Final trade reconciliation (shadow/live depending on flags)
python src/trade_reconciler.py

# Submit orders with lifecycle tracking
python src/execute_order.py

# Evaluate health thresholds
python src/health_monitor.py

# Aggregate latency for dashboard/health
python src/aggregate_latency.py

# Log resolved markets
python src/log_outcomes.py

# Detect cross-venue arbitrage
python src/cross_venue_arb.py

# Counterfactual ensemble tester
python scripts/counterfactual_eval.py

# Quick validation run
python scripts/validate_changes.py
```

### Backtesting & Analysis

```bash
# Portfolio simulation (requires scored + resolved data)
python src/simulate_portfolio.py --bet-sizing fractional_kelly --bankroll 15000

# Stability monitoring & drift detection
python src/stability_monitor.py --update-baseline

# Counterfactual ensemble comparison on current markets
python scripts/counterfactual_eval.py

# Update audit log + produce reports
python src/audit_trail.py
python src/generate_audit_report.py --output-html results/reports/audit_report.html

# Latency aggregation for the dashboard heatmap
python src/aggregate_latency.py

# Launch interactive dashboard
streamlit run src/dashboard.py

# Trade reconciler dry-run
python src/trade_reconciler.py --orders results/portfolio/proposed.json --shadow

# View regime summaries
ls results/regimes/
```

### Helpful Switches

```bash
# Reproduce a run with a fixed seed
python main.py --seed 42

# Only regenerate visuals (expects prior & scored data)
python main.py --visualize-only

# Inspect scored markets interactively with tagging overlays
python src/analyze_opportunities.py --quick --tagging-mode keyword

# Blend keyword + LLM tags for the top 15 markets (requires PERPLEXITY_API_KEY)
python src/analyze_opportunities.py --tagging-mode hybrid --llm-sample-size 15

# Re-run priors with OpenAI/Claude enabled and inspect ensemble history
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python src/generate_priors.py
```

---

## Daily Research Loop

1. `python main.py` — fetch markets, generate ensemble priors, rank mispricings with Beta/LLR logging, and build visuals.
2. `python src/log_outcomes.py --days-back 30` — refresh the resolved/outcome dataset and model scorecards.
3. `python src/generate_priors.py` — re-run priors with drift-aware confidence decay and updated weights (skips markets already scored).
4. `python src/score_mispricing.py` → `python src/signal_evaluator.py` — refresh mispricing ranks, persist Beta/LLR fields, and gate with tagged reasons.
5. `python src/cross_venue_arb.py` — detect and score cross-venue arbitrage opportunities (integrates with scored markets).
6. `python src/simulate_portfolio.py --bet-sizing fractional_kelly` — run a what-if portfolio pass with regime mispricing volatility, non-linear slippage, and drawdown metrics.
7. `python src/trade_reconciler.py --orders orders/pending.json --shadow` — finalize trades with risk limits, de-risking rules, and kill-switch/health context.
8. `python src/execute_order.py --order-file orders/pending.json --shadow-trade` — submit validated trades via the order agent while remaining in shadow mode.
9. `python src/stability_monitor.py --update-baseline` — log calibration/distribution drift and refresh alert baselines feeding confidence decay.
10. `python src/aggregate_latency.py` — roll latency JSONL logs into `results/health/latency_summary.csv` for the dashboard.
11. `python src/health_monitor.py` — evaluate latency/slippage/drawdown velocity thresholds and engage/reset the kill switch as needed.
12. `python scripts/counterfactual_eval.py` — compare the current ensemble to historical weights on the latest markets.
13. Review `results/regimes/` for regime counts + transition stats and inspect `results/portfolio/` drawdown/velocity curves before allocating.
14. `python src/audit_trail.py` → `python src/generate_audit_report.py` to archive the day's work and shareable artifact; `python scripts/validate_changes.py --run-shadow-order` remains an optional end-to-end validation.

### Artifacts to Inspect After a Run

- `results/raw/` — raw Polymarket snapshots (JSON + latest CSV)
- `results/priors/` — prior estimates (`*_priors.csv`, `latest_priors.csv`)
- `results/priors/ensemble_history.jsonl` — per-run ensemble weights by category and drift/decay annotations
- `results/scored/` — scored mispricings (`*_scored_markets.csv`, `latest_scored_markets.csv`, `*_arb_opps.csv`) with Beta/LLR fields, `evaluation_reason`, and `mispricing_vol_regime`
- `results/figures/` — charts, dashboards, and other visuals
- `results/reports/` — text summaries and exports
- `results/logs/` — machine-readable pipeline logs (`{run_id}_*.jsonl`)
- `results/prompts/` — LLM prompt/response archives (`{run_id}_prompts.jsonl`)
- `results/resolved/` — resolved market snapshots + joined calibration datasets
- `results/portfolio/` — portfolio trades, equity curves, drawdown/velocity series, slippage stats, and factor attribution
- `results/regimes/` — regime counts & transition history (`regime_history.csv`, per-run summaries)
- `results/stability/` — drift & calibration reports (`*_stability.json`, `baseline.json`, `counterfactual_eval_*.json`)
- `results/health/` — pipeline latency logs (`latency_logs.jsonl`), summaries (`latency_summary.csv`), health status snapshots, and kill-switch state
- `results/audit/` — per-run trade traces and archived audit reports
- `notebooks/calibration_analysis.ipynb` — Brier/ECE/log-loss diagnostics & meta-analysis starter

---

## Containerization & CI/CD

Build service images (scraper, priors generator, order agent, dashboard):

```bash
docker compose build
docker compose up dashboard  # launches Streamlit on http://localhost:8501
```

The shared `results/` directory is mounted into each container to persist artifacts across services.

GitHub Actions workflow `.github/workflows/ci.yml` runs linting (ruff), bytecode compilation, pytest, and Docker builds for every push/pull request.

---

## Shadow Trading Mode

Enable shadow trading to log intended orders without hitting venue APIs:

```bash
export NAD_SHADOW_TRADE=1
python src/execute_order.py --slug example-market --direction buy_yes --stake 250 --price 0.45
```

The order agent records a `shadow` lifecycle entry in `results/logs/<run>_order_agent.jsonl`, enabling live-vs-sim reconciliation without transmitting orders.

---


### Additional Resources

- **Polymarket Gamma API Docs** — For technical reference on market data structure and reliability.
- **MIT Prediction Market Database** — For historical, resolved outcomes across research and live markets.
- **Kalshi API Documentation** — For public market data and odds fetching (used in cross-venue comparison).
- **PredictIt API** — Public endpoint for all market data, enabling cross-platform arbitrage analysis.
