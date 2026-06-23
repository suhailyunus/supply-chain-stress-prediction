# Dataset Notes

## Dataset

M5 Forecasting Accuracy Dataset

## Initial Questions

* What defines a shortage event?
* Can demand spikes be used as a proxy?
* Which products exhibit the highest volatility?

## Evaluation Ideas

* Temporal split
* Missing data robustness
* Demand shock simulation
* Cross-category generalization

## Initial Stress Event Definition

A stress event occurs when daily sales exceed the item's historical 90th percentile.

### Rationale

* Identifies unusually high demand periods.
* Acts as a proxy for inventory pressure.
* Can be computed consistently across products.
* Creates a reproducible target variable for experimentation.

## Lessons Learned

### Data Representation

The original M5 dataset is stored in a wide format, with one column per day.

To support machine learning workflows, the data was reshaped into a long format using `pandas.melt()`, resulting in one row per product-day observation.

### Scalability Observation

The full transformed dataset consumed approximately 22.8 GB of memory.

To enable rapid experimentation, an initial subset of 100 products was selected, reducing memory usage to approximately 0.75 GB.

### Stress Event Definition

Because the dataset does not contain inventory levels or explicit stockout events, a proxy target is required.

The initial definition of a stress event is:

* Daily sales exceed the item's historical 90th percentile.

This identifies unusually high demand periods that may correspond to inventory pressure.

## Initial Class Balance

After defining stress events as:

sales > item-specific 90th percentile

the resulting class distribution was:

* Normal days (0): 93.9%
* Stress days (1): 6.1%

Observation:

The positive class is relatively rare but still sufficiently represented for experimentation. This suggests the initial threshold produces a meaningful distinction between typical demand and unusually high demand.

## Baseline Logistic Regression Results

Unweighted logistic regression achieved high overall accuracy (94%) but failed to detect stress events (recall ≈ 0).

Applying class balancing reduced overall accuracy to 78% but improved stress-event recall to 30%.

Interpretation:

This highlights the importance of class imbalance handling in supply-chain stress prediction. Traditional accuracy metrics may obscure poor rare-event performance.

## Model Comparison — Baseline Experiments

### Logistic Regression (Unweighted)

Results:

* Accuracy: 94%
* Stress-event recall: ~0%

Observation:

Although overall accuracy was high, the model failed to detect rare stress events. This demonstrates that accuracy alone is not sufficient for evaluating imbalanced supply-chain risk problems.

---

### Logistic Regression (Balanced)

Results:

* Accuracy: 78%
* Stress-event recall: 30%
* Stress-event precision: 10%

Observation:

Applying class weighting significantly improved rare-event detection at the cost of lower overall accuracy. This confirms the importance of explicitly handling class imbalance.

---

### Random Forest (Balanced)

Results:

* Accuracy: 38%
* Stress-event recall: 87%
* Stress-event precision: 8%

Observation:

Random Forest dramatically improved recall, capturing most stress events, but at the expense of many false positives. This indicates a strong preference toward aggressive shortage detection.

---

## Business Tradeoff Interpretation

In supply-chain settings:

* False Positive → unnecessary extra inventory
* False Negative → missed shortage / stockout

False negatives are typically more costly than false positives.

This suggests recall may be a more important optimization target than raw accuracy.

---

## Feature Importance Findings

Random Forest feature importance:

* rolling_mean_7 → 69.4%
* sales_lag_1 → 18.9%
* sales_lag_7 → 11.7%

Interpretation:

Recent rolling demand trends appear to be substantially more predictive of stress events than isolated single-day observations.

This suggests that sustained elevated demand may be a stronger precursor to supply stress than abrupt one-day spikes.

## Volatility Feature Experiment

Added:

* rolling_std_7 (7-day rolling standard deviation)

Purpose:

To measure short-term sales volatility and test whether instability contributes to supply stress.

Results:

Random Forest performance remained largely unchanged:

* Stress recall: 86%
* Stress precision: 8%

However, feature importance changed substantially:

* rolling_mean_7 → 48.0%
* rolling_std_7 → 40.8%
* sales_lag_1 → 9.3%
* sales_lag_7 → 1.9%

Interpretation:

Volatility appears to be a major explanatory feature for stress-event prediction, nearly as important as recent average demand.

This suggests that supply stress may be driven by both sustained demand pressure and demand instability.
