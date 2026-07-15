# Supply Chain Stress Evaluation

This project develops an early-warning classifier for unusually high retail demand using the M5 Forecasting dataset. Because M5 does not contain direct inventory or stockout labels, the project defines an item-relative high-demand proxy for supply stress and evaluates whether recent demand, calendar, pricing, and location signals can identify those events.

## Project architecture

```text
Raw M5 CSV files
      |
      v
src/load_data.py
      |
      v
src/preprocess.py
  - sample development items
  - reshape sales to long form
  - construct stress target
  - merge calendar and prices
      |
      v
src/features.py
  - item-store lags
  - rolling demand statistics
  - calendar indicators
  - price movement
  - geographic encoding
      |
      v
src/train.py / src/pipeline.py
  - chronological validation
  - class-balanced XGBoost
  - artifact persistence
      |
      v
src/predict.py
  - shared feature logic
  - probability scoring
  - business-readable risk labels
```

## Repository layout

```text
.
├── data/raw/                  # M5 source files (not committed)
├── models/                    # Generated model artifacts
├── notebooks/
│   └── supply_stress_case_study.ipynb
├── reports/figures/
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place these files in `data/raw/`:

   - `sales_train_validation.csv`
   - `calendar.csv`
   - `sell_prices.csv`

## Run the training pipeline

From the repository root:

```python
from src.pipeline import run_training_pipeline

result = run_training_pipeline(
    data_dir="data/raw",
    max_items=100,
    models_dir="models",
)
```

## Run inference

Inference requires recent historical rows because lags and rolling features cannot be constructed from an isolated observation.

```python
from src.predict import predict_supply_stress

predictions = predict_supply_stress(
    recent_data,
    models_dir="models",
    threshold=0.50,
)
```

## Methodological notes

- Demand features are grouped by both `item_id` and `store_id`.
- Rolling features use `shift(1)` to avoid current-row leakage.
- The final evaluation uses a chronological holdout.
- Accuracy is not treated as the primary metric because stress events are rare.
- The default operating threshold favors recall for early warning.
- Threshold selection should be performed on a validation period rather than the final test period in a production experiment.

## Current limitations

- The target is a high-demand proxy, not a verified stockout label.
- The development workflow samples the first 100 items for local efficiency.
- Inventory, supplier lead-time, weather, and logistics data are not available in M5.
- Model probabilities may require calibration before operational deployment.
