# gpt2 Training Validation Report

## Goal

Validate whether the training memory estimator generalizes from distilgpt2 to gpt2.

## Experiment Setup

- model = gpt2
- task = training
- batch size = 1
- input tokens = 32, 64, 128
- optimizers = SGD, AdamW
- dtype = fp32
- GPU = Tesla T4 / Colab GPU environment

## Why gpt2 Training Validation Matters

Previous training estimator results were mainly based on distilgpt2.

gpt2 is larger than distilgpt2, so this test checks whether optimizer-state and backward-temporary-memory formulas generalize to a larger GPT-style model.

## gpt2 Training Metrics

| model_name   | phase    |   num_rows |   allocated_MRE |   allocated_mean_error |   allocated_max_error |   allocated_min_error |   reserved_MRE |   reserved_mean_error |   reserved_max_error |   reserved_min_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|:---------|-----------:|----------------:|-----------------------:|----------------------:|----------------------:|---------------:|----------------------:|---------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| gpt2         | training |          6 |         5.02757 |                6.93978 |               17.6936 |               1.96771 |        5.26279 |               6.51771 |              16.8337 |              1.29213 |                   1846.37 |                      1894.43 |                  2022.67 |                     2045.99 |

## Error by Optimizer

| optimizer_name   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-----------------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| adamw            |          3 |                3.08698 |                  2.51955 |               4.77366 |               3.93924 |                 4.98916 |              5.53643 |                   2411.71 |                      2369.13 |                  2640    |                     2558.66 |
| sgd              |          3 |               10.7926  |                  9.40266 |              17.6936  |               9.09619 |                 6.28953 |             16.8337  |                   1281.03 |                      1419.73 |                  1405.33 |                     1533.31 |

## Error by Sequence Length

|   input_tokens |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|---------------:|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
|             32 |          2 |                5.02757 |                  5.02757 |               5.28148 |               4.85087 |                 4.85087 |              5.53643 |                   1837.19 |                      1813.42 |                     2002 |                     1958.49 |
|             64 |          2 |                5.9611  |                  5.9611  |               9.40266 |               5.63934 |                 5.63934 |              6.28953 |                   1844.39 |                      1874.18 |                     2046 |                     2024.11 |
|            128 |          2 |                9.83068 |                  9.83068 |              17.6936  |               9.06293 |                 9.06293 |             16.8337  |                   1857.53 |                      1995.7  |                     2020 |                     2155.36 |

## Model Generalization Comparison

| model_name   | phase    |   allocated_mean_error |   reserved_mean_error |   allocated_MRE |   reserved_MRE | note                                       |
|:-------------|:---------|-----------------------:|----------------------:|----------------:|---------------:|:-------------------------------------------|
| distilgpt2   | training |                3.46251 |               3.86821 |         2.97717 |        3.49684 | from distilgpt2 TrainingMemoryEstimator V2 |
| gpt2         | training |                6.93978 |               6.51771 |         5.02757 |        5.26279 | from gpt2 training validation              |

## Main Interpretation

If gpt2 training error is close to distilgpt2, this supports training estimator generalization.

If gpt2 training error is worse, it suggests the backward temporary correction or allocator correction may need model-size-specific tuning.

## Limitations

- The validation grid is small.
- Only batch size 1 was tested.
- Only fp32 training was tested.
- The experiment uses a single GPU environment.
- More models are needed before claiming broad training-memory generalization.

## Next Step

Use these results to decide whether the training estimator needs model-size-specific tuning.