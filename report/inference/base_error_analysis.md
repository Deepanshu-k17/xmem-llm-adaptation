
# Day 12 - Base Estimator Error Analysis

## Goal

Analyze where the BaseMemoryEstimator works and where it fails.

Day 11 built the first baseline memory estimator. Day 12 focuses on error analysis before improving the estimator.

## Global Metrics

- Global MRE: 6.82%
- Global mean relative error: 42.46%
- Global max relative error: 95.14%
- Global min relative error: 1.05%

## Error by Model

### tiny-gpt2

- Mean relative error: 94.52%
- Median relative error: 95.14%

### distilgpt2

- Mean relative error: 5.28%
- Median relative error: 5.32%

## Main Finding

The BaseMemoryEstimator performs much worse on tiny-gpt2 than on distilgpt2.

The main reason is the fixed overhead term in the estimator.

For distilgpt2, fixed overhead is small relative to total memory.

For tiny-gpt2, actual peak memory is only around 12.5 MB, so a fixed overhead term creates a very large relative error.

## Why MRE and Mean Error Differ

The global MRE is low because median error is not strongly affected by extreme outliers.

The mean error is much higher because tiny-gpt2 rows have very high relative error.

This shows why both median and mean error should be reported.

## Estimator Weaknesses Identified

1. Fixed overhead is too large for tiny models.
2. Token memory component is currently too simple.
3. Allocator behavior is not explicitly modeled.
4. use_cache behavior is not modeled accurately.
5. dtype effects are only handled through parameter memory, not all memory components.

## Next Step

The next step is to improve the estimator.

Possible improvements:

1. Add model-size-aware overhead correction.
2. Add KV-cache-specific memory component.
3. Add precision-aware correction.
4. Add allocator correction to estimate reserved memory.

