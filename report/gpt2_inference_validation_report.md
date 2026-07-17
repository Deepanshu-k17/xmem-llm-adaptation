# gpt2 Inference Validation Report

## Goal

Validate whether the inference memory estimator generalizes from distilgpt2 to gpt2.

## Experiment Setup

- model = gpt2
- task = inference
- batch sizes = 1, 2
- input tokens = 64, 128, 256
- max new tokens = 32, 128
- dtype = fp32, with one fp16 sanity run if supported
- use_cache = True

## Why gpt2 Validation Matters

Most previous meaningful inference results were based on distilgpt2.

gpt2 is larger than distilgpt2, so this validation checks whether the estimator generalizes to a larger GPT-style model.

## gpt2 Estimator Metrics

| model_name   |   num_rows |   allocated_MRE |   allocated_mean_error |   allocated_max_error |   allocated_min_error |   reserved_MRE |   reserved_mean_error |   reserved_max_error |   reserved_min_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|-----------:|----------------:|-----------------------:|----------------------:|----------------------:|---------------:|----------------------:|---------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| gpt2         |         13 |         2.65902 |                2.49934 |               5.09383 |              0.213731 |        2.06186 |               2.28433 |              5.22388 |             0.675676 |                   502.585 |                      504.455 |                  546.615 |                     548.462 |

## Error by dtype

| dtype   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:--------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| fp16    |          1 |                2.01717 |                  2.01717 |               2.01717 |               5.22388 |                 5.22388 |              5.22388 |                   264.78  |                      259.439 |                  268     |                     282     |
| fp32    |         12 |                2.53952 |                  2.97326 |               5.09383 |               2.03937 |                 2.04444 |              3.7037  |                   522.402 |                      524.873 |                  569.833 |                     570.667 |

## Error by batch size

|   batch_size |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|-------------:|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
|            1 |          7 |                2.56022 |                  2.65902 |               4.3083  |               2.85155 |                 2.54545 |              5.22388 |                    474.08 |                      482.617 |                      518 |                     524.857 |
|            2 |          6 |                2.42831 |                  2.17272 |               5.09383 |               1.62258 |                 1.7008  |              2.90909 |                    535.84 |                      529.932 |                      580 |                     576     |

## Model Generalization Comparison

| model_name   | phase     |   allocated_mean_error |   reserved_mean_error | note                                          |
|:-------------|:----------|-----------------------:|----------------------:|:----------------------------------------------|
| distilgpt2   | inference |                2.36    |               2.36    | from previous distilgpt2 inference evaluation |
| gpt2         | inference |                2.49934 |               2.28433 | from gpt2 validation                          |

## Main Interpretation

The gpt2 validation checks whether the estimator can handle a larger GPT-style model beyond distilgpt2.

If gpt2 error is close to distilgpt2 error, this supports model generalization.

If gpt2 error is much higher, it means the estimator needs model-size-specific correction.

## Limitations

- The validation grid is intentionally small.
- The experiment uses a single GPU environment.
- Only GPT-style causal language models are tested.
- More models are needed before claiming broad generalization.

## Next Step

Run limited gpt2 training validation and compare training estimator behavior across distilgpt2 and gpt2.