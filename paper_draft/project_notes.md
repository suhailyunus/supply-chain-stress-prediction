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

