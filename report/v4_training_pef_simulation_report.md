# V4 Training PEF-style Fit/Fail Simulation Report

## 1. Goal

This report evaluates whether TrainingMemoryEstimator V4 can correctly predict whether a training workload will fit under a given GPU memory limit.

The PEF-style simulation converts memory prediction into a deployment-style fit/fail decision.

## 2. Method

For each training workload and memory limit, the simulation compares actual and predicted peak reserved memory.

`actual_fits = actual_peak_reserved_MB <= memory_limit_MB`

`predicted_fits = predicted_peak_reserved_MB <= memory_limit_MB`

Each case is classified as:

- correct
- dangerous underprediction
- conservative overprediction

A dangerous underprediction means the estimator predicts the workload will fit, but it actually does not fit.

A conservative overprediction means the estimator predicts the workload will not fit, but it actually would fit.

## 3. Memory Limits Tested

768, 1024, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2048, 2200, 2400, 2600, 2800, 3072, 4096, 8192

## 4. Global PEF Summary

|   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
|           266 |             256 |                    8 |                       2 |            96.2406 |                          3.00752 |                             0.75188 |

## 5. PEF by Memory Limit

|   memory_limit_MB |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|------------------:|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
|               768 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              1024 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|              1200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              1300 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|              1400 |            14 |              13 |                    1 |                       0 |            92.8571 |                          7.14286 |                             0       |
|              1500 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              1600 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              1700 |            14 |              11 |                    3 |                       0 |            78.5714 |                         21.4286  |                             0       |
|              1800 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|              1900 |            14 |              13 |                    0 |                       1 |            92.8571 |                          0       |                             7.14286 |
|              2000 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              2048 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              2200 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              2400 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              2600 |            14 |              12 |                    2 |                       0 |            85.7143 |                         14.2857  |                             0       |
|              2800 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              3072 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              4096 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |
|              8192 |            14 |              14 |                    0 |                       0 |           100      |                          0       |                             0       |

## 6. PEF by Model

| model_name   |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|:-------------|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
| distilgpt2   |           152 |             146 |                    4 |                       2 |            96.0526 |                          2.63158 |                             1.31579 |
| gpt2         |           114 |             110 |                    4 |                       0 |            96.4912 |                          3.50877 |                             0       |

## 7. PEF by Optimizer

| optimizer_name   |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|:-----------------|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
| adamw            |           190 |             183 |                    5 |                       2 |            96.3158 |                          2.63158 |                             1.05263 |
| sgd              |            76 |              73 |                    3 |                       0 |            96.0526 |                          3.94737 |                             0       |

## 8. Dangerous Failures

| model_name   |   batch_size |   input_tokens | dtype   | optimizer_name   |   memory_limit_MB |   actual_peak_allocated_MB |   predicted_peak_allocated_MB |   actual_peak_reserved_MB |   predicted_peak_reserved_MB | actual_fits   | predicted_fits   | correct_prediction   | dangerous_failure   | conservative_failure   | outcome                   |
|:-------------|-------------:|---------------:|:--------|:-----------------|------------------:|---------------------------:|------------------------------:|--------------------------:|-----------------------------:|:--------------|:-----------------|:---------------------|:--------------------|:-----------------------|:--------------------------|
| distilgpt2   |            1 |             64 | fp32    | adamw            |              1700 |                    1600.69 |                      1552.99  |                      1738 |                      1677.22 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| distilgpt2   |            1 |             64 | fp32    | adamw            |              1700 |                    1600.19 |                      1552.99  |                      1738 |                      1677.22 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| distilgpt2   |            1 |             64 | fp32    | sgd              |              1024 |                     956.49 |                       928.043 |                      1030 |                      1002.29 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| distilgpt2   |            1 |             64 | fp32    | adamw            |              1700 |                    1600.94 |                      1552.99  |                      1738 |                      1677.22 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| gpt2         |            1 |             32 | fp32    | sgd              |              1300 |                    1271.56 |                      1196.31  |                      1388 |                      1292.01 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| gpt2         |            1 |             64 | fp32    | sgd              |              1400 |                    1279.2  |                      1257.07  |                      1422 |                      1357.63 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| gpt2         |            1 |             32 | fp32    | adamw            |              2600 |                    2402.82 |                      2288.12  |                      2616 |                      2471.17 | False         | True             | False                | True                | False                  | dangerous_underprediction |
| gpt2         |            1 |             64 | fp32    | adamw            |              2600 |                    2409.59 |                      2348.88  |                      2670 |                      2536.79 | False         | True             | False                | True                | False                  | dangerous_underprediction |

## 9. Conservative Failures

| model_name   |   batch_size |   input_tokens | dtype   | optimizer_name   |   memory_limit_MB |   actual_peak_allocated_MB |   predicted_peak_allocated_MB |   actual_peak_reserved_MB |   predicted_peak_reserved_MB | actual_fits   | predicted_fits   | correct_prediction   | dangerous_failure   | conservative_failure   | outcome                     |
|:-------------|-------------:|---------------:|:--------|:-----------------|------------------:|---------------------------:|------------------------------:|--------------------------:|-----------------------------:|:--------------|:-----------------|:---------------------|:--------------------|:-----------------------|:----------------------------|
| distilgpt2   |            4 |             64 | fp32    | adamw            |              1800 |                    1648.43 |                       1792.96 |                      1768 |                       1936.4 | True          | False            | False                | False               | True                   | conservative_overprediction |
| distilgpt2   |            4 |             64 | fp32    | adamw            |              1900 |                    1648.43 |                       1792.96 |                      1768 |                       1936.4 | True          | False            | False                | False               | True                   | conservative_overprediction |

## 10. Interpretation

PEF-style evaluation is stricter than MRE because small numeric errors near a memory boundary can change a fit/fail decision.

Dangerous failures are more important than conservative failures because they can cause OOM during scheduling.

Failures are expected to occur mostly near tight memory limits where actual and predicted reserved memory are close to the boundary.

## 11. Next Step

The next step is to run a safety-margin simulation on V4 predictions.

The goal is to check whether adding a small margin to predicted reserved memory can remove dangerous failures while keeping conservative failures reasonable.