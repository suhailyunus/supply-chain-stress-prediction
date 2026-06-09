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
