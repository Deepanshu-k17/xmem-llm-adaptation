
# Day 29 - Training PEF-style Fit/Fail Simulation

## Goal

Evaluate whether TrainingMemoryEstimator V2 can predict whether a training workload fits inside a given GPU memory limit.

This moves beyond numeric prediction error and tests scheduling-style fit/fail behavior.

## Input Data

The simulation used Day 28 TrainingMemoryEstimator V2 predictions.

The key fields were:

- actual_peak_reserved_MB
- predicted_peak_reserved_MB

Reserved memory was used because it better reflects PyTorch CUDA allocator behavior and real memory pressure.

## Memory Limits Tested

[512, 768, 1024, 1200, 1400, 1500, 1600, 1700, 1725, 1750, 1800, 2048, 4096, 8192]

## Definitions

### Correct Prediction

The prediction is correct when actual_fits and predicted_fits match.

### Dangerous Failure

Predicted fits, but actual does not fit.

This can cause OOM.

### Conservative Failure

Predicted does not fit, but actual fits.

This wastes capacity but is safer than dangerous underprediction.

## Overall Summary

 total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
         112            102                   7                      3         91.071429                            6.25                           2.678571

## Summary by Memory Limit

 memory_limit_MB  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
             512            8              8                   0                      0             100.0                             0.0                                0.0
             768            8              8                   0                      0             100.0                             0.0                                0.0
            1024            8              7                   1                      0              87.5                            12.5                                0.0
            1200            8              8                   0                      0             100.0                             0.0                                0.0
            1400            8              8                   0                      0             100.0                             0.0                                0.0
            1500            8              8                   0                      0             100.0                             0.0                                0.0
            1600            8              8                   0                      0             100.0                             0.0                                0.0
            1700            8              5                   3                      0              62.5                            37.5                                0.0
            1725            8              5                   3                      0              62.5                            37.5                                0.0
            1750            8              6                   0                      2              75.0                             0.0                               25.0
            1800            8              7                   0                      1              87.5                             0.0                               12.5
            2048            8              8                   0                      0             100.0                             0.0                                0.0
            4096            8              8                   0                      0             100.0                             0.0                                0.0
            8192            8              8                   0                      0             100.0                             0.0                                0.0

## Summary by Optimizer

optimizer_name  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
         adamw           98             89                   6                      3         90.816327                        6.122449                           3.061224
           sgd           14             13                   1                      0         92.857143                        7.142857                           0.000000

## Summary by Experiment Type

                     experiment_type  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
     distilgpt2_batch_size_variation           42             38                   2                      2         90.476190                        4.761905                           4.761905
     distilgpt2_optimizer_comparison           28             25                   3                      0         89.285714                       10.714286                           0.000000
distilgpt2_sequence_length_variation           42             39                   2                      1         92.857143                        4.761905                           2.380952

## Dangerous Failures

model_name  batch_size  input_tokens optimizer_name                      experiment_type  memory_limit_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_fits  predicted_fits  correct_prediction  dangerous_failure  conservative_failure                   outcome
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation             1700                   1738.0                 1677.224945                   1600.69                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation             1725                   1738.0                 1677.224945                   1600.69                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation             1700                   1738.0                 1677.224945                   1600.19                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation             1725                   1738.0                 1677.224945                   1600.19                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64            sgd      distilgpt2_optimizer_comparison             1024                   1030.0                 1002.286117                    956.49                   928.042701        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison             1700                   1738.0                 1677.224945                   1600.94                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison             1725                   1738.0                 1677.224945                   1600.94                  1552.986061        False            True               False               True                 False dangerous_underprediction

## Conservative Failures

model_name  batch_size  input_tokens optimizer_name                      experiment_type  memory_limit_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_fits  predicted_fits  correct_prediction  dangerous_failure  conservative_failure                     outcome
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation             1750                   1726.0                 1763.617115                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation             1750                   1726.0                 1763.617115                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation             1800                   1768.0                 1936.401455                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction

## Realistic GPU Limit Summary

 memory_limit_MB  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent
            2048            8              8                   0                      0             100.0
            4096            8              8                   0                      0             100.0
            8192            8              8                   0                      0             100.0

## Main Interpretation

The purpose of this simulation is to evaluate practical fit/fail prediction behavior.

Dangerous failures are worse than conservative failures because they can cause OOM.

A useful training memory estimator should minimize dangerous failures while keeping conservative failures reasonable.

## Next Step

Use these results in the training phase report and then continue with model validation.

