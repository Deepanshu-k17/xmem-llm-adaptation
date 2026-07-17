# V4 Training PEF Safety-margin Report

## 1. Goal

This report evaluates whether adding a safety margin to V4 predicted reserved memory can reduce dangerous underpredictions in training fit/fail decisions.

## 2. Motivation

Day 37 showed that TrainingMemoryEstimator V4 achieved 96.24% fit/fail accuracy across 266 training placement cases.

However, it still produced 8 dangerous underpredictions.

Since dangerous underprediction can cause OOM during scheduling, a safety margin is needed.

## 3. Method

The simulation adds a safety margin to predicted reserved memory:

`safe_predicted_reserved_MB = predicted_peak_reserved_MB × (1 + safety_margin)`

The fit/fail decision is then made using the safe predicted value.

## 4. Safety Margins Tested

0.0%, 2.0%, 5.0%, 7.5%, 10.0%, 15.0%

## 5. Safety-margin Summary

|   safety_margin |   safety_margin_percent |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|----------------:|------------------------:|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
|           0     |                     0   |           266 |             256 |                    8 |                       2 |            96.2406 |                          3.00752 |                             0.75188 |
|           0.02  |                     2   |           266 |             259 |                    4 |                       3 |            97.3684 |                          1.50376 |                             1.12782 |
|           0.05  |                     5   |           266 |             257 |                    1 |                       8 |            96.6165 |                          0.37594 |                             3.00752 |
|           0.075 |                     7.5 |           266 |             253 |                    0 |                      13 |            95.1128 |                          0       |                             4.88722 |
|           0.1   |                    10   |           266 |             250 |                    0 |                      16 |            93.985  |                          0       |                             6.01504 |
|           0.15  |                    15   |           266 |             239 |                    0 |                      27 |            89.8496 |                          0       |                            10.1504  |

## 6. Best Safety Margin

|   safety_margin |   safety_margin_percent |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|----------------:|------------------------:|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
|           0.075 |                     7.5 |           266 |             253 |                    0 |                      13 |            95.1128 |                                0 |                             4.88722 |

## 7. Failures at Best Margin

|   safety_margin |   safety_margin_percent | model_name   |   batch_size |   input_tokens | dtype   | optimizer_name   |   memory_limit_MB |   actual_peak_allocated_MB |   predicted_peak_allocated_MB |   actual_peak_reserved_MB |   predicted_peak_reserved_MB |   safe_predicted_reserved_MB | actual_fits   | predicted_fits   | correct_prediction   | dangerous_failure   | conservative_failure   | outcome                     |
|----------------:|------------------------:|:-------------|-------------:|---------------:|:--------|:-----------------|------------------:|---------------------------:|------------------------------:|--------------------------:|-----------------------------:|-----------------------------:|:--------------|:-----------------|:---------------------|:--------------------|:-----------------------|:----------------------------|
|           0.075 |                     7.5 | distilgpt2   |            1 |             32 | fp32    | adamw            |              1700 |                    1592.43 |                       1512.99 |                      1700 |                      1634.03 |                      1756.58 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            1 |             64 | fp32    | adamw            |              1800 |                    1600.69 |                       1552.99 |                      1738 |                      1677.22 |                      1803.02 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            1 |            128 | fp32    | adamw            |              1800 |                    1616.71 |                       1632.98 |                      1726 |                      1763.62 |                      1895.89 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            1 |             64 | fp32    | adamw            |              1800 |                    1600.19 |                       1552.99 |                      1738 |                      1677.22 |                      1803.02 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            2 |             64 | fp32    | adamw            |              1800 |                    1616.21 |                       1632.98 |                      1726 |                      1763.62 |                      1895.89 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            4 |             64 | fp32    | adamw            |              1800 |                    1648.43 |                       1792.96 |                      1768 |                      1936.4  |                      2081.63 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            4 |             64 | fp32    | adamw            |              1900 |                    1648.43 |                       1792.96 |                      1768 |                      1936.4  |                      2081.63 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            4 |             64 | fp32    | adamw            |              2000 |                    1648.43 |                       1792.96 |                      1768 |                      1936.4  |                      2081.63 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            4 |             64 | fp32    | adamw            |              2048 |                    1648.43 |                       1792.96 |                      1768 |                      1936.4  |                      2081.63 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | distilgpt2   |            1 |             64 | fp32    | adamw            |              1800 |                    1600.94 |                       1552.99 |                      1738 |                      1677.22 |                      1803.02 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | gpt2         |            1 |            128 | fp32    | sgd              |              1500 |                    1292.34 |                       1378.59 |                      1406 |                      1488.88 |                      1600.55 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | gpt2         |            1 |            128 | fp32    | sgd              |              1600 |                    1292.34 |                       1378.59 |                      1406 |                      1488.88 |                      1600.55 | True          | False            | False                | False               | True                   | conservative_overprediction |
|           0.075 |                     7.5 | gpt2         |            1 |            128 | fp32    | adamw            |              2800 |                    2422.73 |                       2470.4  |                      2634 |                      2668.03 |                      2868.14 | True          | False            | False                | False               | True                   | conservative_overprediction |

## 8. Safety Margin by Memory Limit

|   safety_margin_percent |   memory_limit_MB |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|------------------------:|------------------:|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
|                     0   |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              1024 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     0   |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              1300 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     0   |              1400 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     0   |              1500 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              1600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              1700 |            14 |              11 |                    3 |                       0 |            78.5714 |                         21.4286  |                             0       |
|                     0   |              1800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     0   |              1900 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     0   |              2000 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              2048 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              2600 |            14 |              12 |                    2 |                       0 |            85.7143 |                         14.2857  |                             0       |
|                     0   |              2800 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     0   |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              1024 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     2   |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              1300 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              1400 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     2   |              1500 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     2   |              1600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              1700 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              1800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     2   |              1900 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     2   |              2000 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              2048 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              2600 |            14 |              12 |                    2 |                       0 |            85.7143 |                         14.2857  |                             0       |
|                     2   |              2800 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     2   |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1024 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1300 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1500 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     5   |              1600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              1700 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     5   |              1800 |            14 |              11 |                    0 |                       3 |            78.5714 |                          0       |                            21.4286  |
|                     5   |              1900 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     5   |              2000 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     5   |              2048 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              2600 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|                     5   |              2800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     5   |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     5   |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              1024 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              1300 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              1400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              1500 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              1600 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              1700 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              1800 |            14 |               8 |                    0 |                       6 |            57.1429 |                          0       |                            42.8571  |
|                     7.5 |              1900 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              2000 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              2048 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              2600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              2800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                     7.5 |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                     7.5 |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              1024 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              1300 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              1400 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              1500 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              1600 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              1700 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              1800 |            14 |               8 |                    0 |                       6 |            57.1429 |                          0       |                            42.8571  |
|                    10   |              1900 |            14 |              11 |                    0 |                       3 |            78.5714 |                          0       |                            21.4286  |
|                    10   |              2000 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              2048 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              2600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              2800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    10   |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    10   |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              1024 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              1300 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              1400 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    15   |              1500 |            14 |              12 |                    0 |                       2 |            85.7143 |                          0       |                            14.2857  |
|                    15   |              1600 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    15   |              1700 |            14 |              12 |                    0 |                       2 |            85.7143 |                          0       |                            14.2857  |
|                    15   |              1800 |            14 |               7 |                    0 |                       7 |            50      |                          0       |                            50       |
|                    15   |              1900 |            14 |               8 |                    0 |                       6 |            57.1429 |                          0       |                            42.8571  |
|                    15   |              2000 |            14 |              11 |                    0 |                       3 |            78.5714 |                          0       |                            21.4286  |
|                    15   |              2048 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    15   |              2200 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|                    15   |              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              2600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              2800 |            14 |              11 |                    0 |                       3 |            78.5714 |                          0       |                            21.4286  |
|                    15   |              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|                    15   |              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |

## 9. Interpretation

The selected best safety margin is 7.50%.

At this margin, accuracy is 95.11%, dangerous failures are 0, and conservative failures are 13.

Safety margin improves placement safety by reducing dangerous underprediction.

The tradeoff is that conservative overprediction may increase, which can waste GPU capacity.

## 10. Conclusion

Safety margin is necessary because even a low-error estimator can make unsafe decisions near memory limits.

V4 should be used with a safety margin when the goal is safe workload placement rather than only low MRE.