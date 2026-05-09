# UFC Fight Prediction System

Production-grade MMA fight prediction system using multi-source data, engineered features, and stacked ensemble models.

## Predictions

- **Fight winner** — calibrated probability with 90% confidence interval
- **Method of victory** — KO/TKO, Submission, Decision distribution
- **Round of finish** — R1-R5 or Decision
- **Fight duration** — seconds (regression for prop bets)

## Architecture

```
Data Sources → Feature Engineering → Stacked Ensemble → Calibrated Predictions
     │                 │                    │
     ├─ UFCStats       ├─ Elo/Glicko-2     ├─ LightGBM (monotonic constraints)
     ├─ Sherdog        ├─ Rolling Stats     ├─ XGBoost
     ├─ Tapology       ├─ Style Matchup     ├─ Neural Net (fighter embeddings)
     ├─ BestFightOdds  ├─ Contextual        └─ Logistic Meta-Learner
     └─ Kaggle         └─ Market Odds
```

## Quick Start

```bash
# Install
pip install -e .

# Scrape data
ufc-predict scrape --source all

# Build features (chronological, no leakage)
ufc-predict build-features --data data/raw/ufcstats_fights.csv --output data/processed/features.parquet

# Train all models
ufc-predict train --features data/processed/features.parquet --save-model models/pipeline.pkl

# Predict a fight
ufc-predict predict \
  --fighter-a "Islam Makhachev" \
  --fighter-b "Charles Oliveira" \
  --date 2026-06-01 \
  --rounds 5 \
  --title-fight \
  --closing-line 0.72

# Full pipeline (scrape → features → train → evaluate)
ufc-predict run-full --data data/raw/ufcstats_fights.csv
```

## Feature Tiers (by expected signal)

| Tier | Category | Features |
|------|----------|----------|
| T1 | Market | Closing line (devigged), opening→closing movement, book consensus |
| T1 | Ratings | Elo, Glicko-2, method-specific Elo, division Elo |
| T1 | Age | Age at fight, age delta, career mileage (strikes/KDs absorbed) |
| T2 | Stats | EWM rolling stats (SLpM, TD avg, Sub avg, defenses) |
| T2 | Matchup | Style cluster interaction, stance edge, reach×style |
| T2 | Form | Win/loss streak, performance trend, quality-adjusted schedule |
| T3 | Context | Layoff, weight misses, camp changes, short notice |
| T3 | Venue | Altitude, octagon size, home advantage |
| T4 | Referee | Early/late stoppage tendencies |

## Anti-Leakage Rules

All features computed using **only pre-fight data**:
- Elo/Glicko update chronologically (not batch)
- Stats use `as_of_date` filtering
- Closing odds are pre-fight (valid features)
- Symmetry verified: swap A↔B → prediction inverts

## Evaluation Benchmarks

The system evaluates against:
1. Closing-line favorite (the real benchmark)
2. Elo-only model
3. Always-pick-favorite baseline
4. Coin flip

Metrics: Log loss (primary), Brier score, accuracy, ECE, and **ROI simulation** (flat $100 bet on >5% edge vs closing line).

## Project Structure

```
ufc_predict/
├── scrapers/          # Data collection from UFCStats, Sherdog, Tapology, BFO
├── features/
│   ├── elo.py         # Elo + Glicko-2 (standard, method-specific, division)
│   ├── stats.py       # Recency-weighted rolling fighter statistics
│   ├── matchup.py     # Style matchup, physical comparisons
│   ├── contextual.py  # Age, layoff, venue, referee, weight cuts
│   └── pipeline.py    # Orchestrates feature generation (temporal ordering)
├── models/
│   ├── gradient_boost.py  # LightGBM + XGBoost with calibration
│   ├── neural.py          # Fighter embedding neural net
│   └── ensemble.py        # Stacked meta-learner
├── evaluation/
│   ├── metrics.py     # Log loss, Brier, accuracy, ECE
│   ├── calibration.py # Calibration curves and reliability diagrams
│   ├── roi.py         # Betting ROI simulation (flat bet + Kelly)
│   └── report.py      # Full evaluation report generation
├── data_loader.py     # Multi-source data merging and deduplication
├── pipeline.py        # End-to-end training and prediction pipeline
├── cli.py             # Click-based CLI interface
├── config.py          # YAML configuration management
└── utils.py           # Shared utilities (scraping, odds conversion, etc.)
```

## Known Limitations

- Model accuracy is bounded by closing-line efficiency (~65-68% accuracy ceiling for MMA)
- Neural net requires sufficient fight history per fighter for meaningful embeddings
- Sherdog/Tapology scraping may require adapting selectors as sites change
- Short-notice replacement detection relies on text patterns (can miss some)
- Pre-UFC fight stats from Sherdog are not as granular as UFCStats data

## Configuration

All hyperparameters in `configs/default.yaml`:
- Time splits, model params, Elo K-factors
- Feature engineering windows and decay rates
- Evaluation thresholds and bet sizing

## Development

```bash
# Run tests
pytest tests/ -v

# Run as module
python -m ufc_predict --help
```
