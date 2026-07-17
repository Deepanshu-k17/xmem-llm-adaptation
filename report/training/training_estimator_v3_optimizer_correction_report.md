# Training Estimator V3 - Optimizer-specific Correction Report

## Goal

Improve the training memory estimator by adding optimizer-specific backward temporary memory correction.

Day 33 showed that TrainingMemoryEstimator V2 generalized reasonably to gpt2, but had significantly higher error for SGD than AdamW.

## Motivation

TrainingMemoryEstimator V2 used the same backward temporary factor for all optimizers:

backward_temp_memory = parameter_memory x 0.65

This worked well for AdamW but overpredicted SGD memory, especially at longer sequence lengths.

## V3 Change

TrainingMemoryEstimator V3 uses optimizer-specific backward temporary factors:

| optimizer | backward_temp_factor |
|---|---:|
| AdamW | 0.65 |
| Adam | 0.65 |
| SGD | 0.35 |
| SGD with momentum | 0.45 |
| default | 0.50 |

## V2 vs V3 Metrics

| estimator                 |   num_rows |   allocated_MRE |   allocated_mean_error |   allocated_max_error |   allocated_min_error |   reserved_MRE |   reserved_mean_error |   reserved_max_error |   reserved_min_error |
|:--------------------------|-----------:|----------------:|-----------------------:|----------------------:|----------------------:|---------------:|----------------------:|---------------------:|---------------------:|
| TrainingMemoryEstimatorV2 |         14 |         2.98779 |                4.95277 |               17.6936 |               1.00629 |        3.68875 |               5.00371 |              16.8337 |              1.29213 |
| TrainingMemoryEstimatorV3 |         14 |         2.98779 |                4.36313 |               12.7747 |               1.00629 |        4.20354 |               4.99495 |              12.5199 |              1.29213 |

## V3 Error by Model

| model_name   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| distilgpt2   |          8 |                4.68758 |                  2.98779 |              12.7747  |               5.09686 |                 3.49684 |             12.5199  |                   1529.01 |                      1508.15 |                  1645.5  |                     1628.8  |
| gpt2         |          6 |                3.93054 |                  3.64661 |               6.67409 |               4.85907 |                 5.26279 |              6.91559 |                   1846.37 |                      1823.23 |                  2022.67 |                     1969.09 |

## V3 Error by Optimizer

| optimizer_name   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-----------------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| adamw            |         10 |                3.39868 |                  2.96505 |                8.768  |               4.00727 |                 3.49684 |              9.52497 |                   1851.07 |                      1833.83 |                   2005.4 |                     1980.53 |
| sgd              |          4 |                6.77426 |                  6.29612 |               12.7747 |               7.46414 |                 6.40514 |             12.5199  |                   1199.9  |                      1166.57 |                   1311.5 |                     1259.89 |

## V3 Error by Sequence Length

| model_name   |   input_tokens |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|---------------:|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| distilgpt2   |             32 |          1 |                4.98862 |                  4.98862 |               4.98862 |               3.88066 |                 3.88066 |              3.88066 |                   1592.43 |                      1512.99 |                     1700 |                     1634.03 |
| distilgpt2   |             64 |          6 |                5.25095 |                  2.98779 |              12.7747  |               5.7858  |                 3.49684 |             12.5199  |                   1503.83 |                      1486.53 |                     1623 |                     1605.46 |
| distilgpt2   |            128 |          1 |                1.00629 |                  1.00629 |               1.00629 |               2.17944 |                 2.17944 |              2.17944 |                   1616.71 |                      1632.98 |                     1726 |                     1763.62 |
| gpt2         |             32 |          2 |                5.34591 |                  5.34591 |               5.91816 |               6.22601 |                 6.22601 |              6.91559 |                   1837.19 |                      1742.21 |                     2002 |                     1881.59 |
| gpt2         |             64 |          2 |                2.12482 |                  2.12482 |               2.51955 |               4.75779 |                 4.75779 |              4.98916 |                   1844.39 |                      1802.97 |                     2046 |                     1947.21 |
| gpt2         |            128 |          2 |                4.3209  |                  4.3209  |               6.67409 |               3.59341 |                 3.59341 |              5.89469 |                   1857.53 |                      1924.5  |                     2020 |                     2078.46 |

## Row-wise V2 vs V3 Comparison

| model_name   |   batch_size |   input_tokens | dtype   | optimizer_name   |   actual_peak_allocated_MB |   actual_peak_reserved_MB |   v2_predicted_allocated_MB |   v2_allocated_error_percent |   v2_predicted_reserved_MB |   v2_reserved_error_percent |   v3_predicted_allocated_MB |   v3_allocated_error_percent |   v3_predicted_reserved_MB |   v3_reserved_error_percent |   backward_temp_factor |   allocated_error_improvement_percent_points |   reserved_error_improvement_percent_points |
|:-------------|-------------:|---------------:|:--------|:-----------------|---------------------------:|--------------------------:|----------------------------:|-----------------------------:|---------------------------:|----------------------------:|----------------------------:|-----------------------------:|---------------------------:|----------------------------:|-----------------------:|---------------------------------------------:|--------------------------------------------:|
| distilgpt2   |            1 |             32 | fp32    | adamw            |                    1592.43 |                      1700 |                    1512.99  |                      4.98862 |                    1634.03 |                     3.88066 |                    1512.99  |                      4.98862 |                   1634.03  |                     3.88066 |                   0.65 |                                  0           |                                 0           |
| distilgpt2   |            1 |             64 | fp32    | adamw            |                    1600.69 |                      1738 |                    1552.99  |                      2.98021 |                    1677.22 |                     3.49684 |                    1552.99  |                      2.98021 |                   1677.22  |                     3.49684 |                   0.65 |                                  0           |                                 0           |
| distilgpt2   |            1 |            128 | fp32    | adamw            |                    1616.71 |                      1726 |                    1632.98  |                      1.00629 |                    1763.62 |                     2.17944 |                    1632.98  |                      1.00629 |                   1763.62  |                     2.17944 |                   0.65 |                                 -2.22045e-16 |                                 0           |
| distilgpt2   |            1 |             64 | fp32    | adamw            |                    1600.19 |                      1738 |                    1552.99  |                      2.9499  |                    1677.22 |                     3.49684 |                    1552.99  |                      2.9499  |                   1677.22  |                     3.49684 |                   0.65 |                                 -4.44089e-16 |                                 0           |
| distilgpt2   |            2 |             64 | fp32    | adamw            |                    1616.21 |                      1726 |                    1632.98  |                      1.03754 |                    1763.62 |                     2.17944 |                    1632.98  |                      1.03754 |                   1763.62  |                     2.17944 |                   0.65 |                                  2.22045e-16 |                                 0           |
| distilgpt2   |            4 |             64 | fp32    | adamw            |                    1648.43 |                      1768 |                    1792.96  |                      8.768   |                    1936.4  |                     9.52497 |                    1792.96  |                      8.768   |                   1936.4   |                     9.52497 |                   0.65 |                                  0           |                                -1.77636e-15 |
| distilgpt2   |            1 |             64 | fp32    | sgd              |                     956.49 |                      1030 |                     928.043 |                      2.97413 |                    1002.29 |                     2.69067 |                     834.301 |                     12.7747  |                    901.045 |                    12.5199  |                   0.35 |                                 -9.80057     |                                -9.82921     |
| distilgpt2   |            1 |             64 | fp32    | adamw            |                    1600.94 |                      1738 |                    1552.99  |                      2.99536 |                    1677.22 |                     3.49684 |                    1552.99  |                      2.99536 |                   1677.22  |                     3.49684 |                   0.65 |                                  0           |                                 0           |
| gpt2         |            1 |             32 | fp32    | sgd              |                    1271.56 |                      1388 |                    1338.72  |                      5.28148 |                    1445.81 |                     4.16531 |                    1196.31  |                      5.91816 |                   1292.01  |                     6.91559 |                   0.35 |                                 -0.63668     |                                -2.75028     |
| gpt2         |            1 |             64 | fp32    | sgd              |                    1279.2  |                      1422 |                    1399.48  |                      9.40266 |                    1511.44 |                     6.28953 |                    1257.07  |                      1.73009 |                   1357.63  |                     4.52643 |                   0.35 |                                  7.67257     |                                 1.7631      |
| gpt2         |            1 |            128 | fp32    | sgd              |                    1292.34 |                      1406 |                    1521     |                     17.6936  |                    1642.68 |                    16.8337  |                    1378.59  |                      6.67409 |                   1488.88  |                     5.89469 |                   0.35 |                                 11.0196      |                                10.939       |
| gpt2         |            1 |             32 | fp32    | adamw            |                    2402.82 |                      2616 |                    2288.12  |                      4.77366 |                    2471.17 |                     5.53643 |                    2288.12  |                      4.77366 |                   2471.17  |                     5.53643 |                   0.65 |                                  0           |                                 0           |
| gpt2         |            1 |             64 | fp32    | adamw            |                    2409.59 |                      2670 |                    2348.88  |                      2.51955 |                    2536.79 |                     4.98916 |                    2348.88  |                      2.51955 |                   2536.79  |                     4.98916 |                   0.65 |                                  0           |                                 0           |
| gpt2         |            1 |            128 | fp32    | adamw            |                    2422.73 |                      2634 |                    2470.4   |                      1.96771 |                    2668.03 |                     1.29213 |                    2470.4   |                      1.96771 |                   2668.03  |                     1.29213 |                   0.65 |                                  0           |                                -2.22045e-16 |

## Main Interpretation

V3 tests whether optimizer-specific correction improves training memory prediction across distilgpt2 and gpt2.

If V3 reduces SGD error without damaging AdamW error too much, this supports optimizer-specific correction.

If V3 improves gpt2 but damages distilgpt2, the correction may need model-specific tuning.

## Next Step

Use V3 results to decide whether the final estimator should use optimizer-specific correction or keep the simpler V2 formula.