
# Day 30 - Training PEF Safety-margin Simulation

## Goal

Evaluate whether adding a safety margin to predicted reserved memory reduces dangerous fit/fail failures.

Day 29 showed that TrainingMemoryEstimator V2 achieved good overall PEF accuracy, but still had dangerous failures near tight memory limits.

## Safety-margin Method

safe_predicted_reserved_MB =
predicted_peak_reserved_MB x (1 + safety_margin)

Safety margins tested:

[0.0, 0.05, 0.1, 0.15]

## Summary by Safety Margin

 safety_margin  safety_margin_percent  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
          0.00                    0.0          112            102                   7                      3         91.071429                            6.25                           2.678571
          0.05                    5.0          112            103                   0                      9         91.964286                            0.00                           8.035714
          0.10                   10.0          112             97                   0                     15         86.607143                            0.00                          13.392857
          0.15                   15.0          112             96                   0                     16         85.714286                            0.00                          14.285714

## Best Safety Margin Candidate

 safety_margin  safety_margin_percent  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
          0.05                    5.0        112.0          103.0                 0.0                    9.0         91.964286                             0.0                           8.035714

## Summary by Memory Limit

 safety_margin_percent  memory_limit_MB  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
                   0.0              512            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0              768            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             1024            8              7                   1                      0              87.5                            12.5                                0.0
                   0.0             1200            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             1400            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             1500            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             1600            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             1700            8              5                   3                      0              62.5                            37.5                                0.0
                   0.0             1725            8              5                   3                      0              62.5                            37.5                                0.0
                   0.0             1750            8              6                   0                      2              75.0                             0.0                               25.0
                   0.0             1800            8              7                   0                      1              87.5                             0.0                               12.5
                   0.0             2048            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             4096            8              8                   0                      0             100.0                             0.0                                0.0
                   0.0             8192            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0              512            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0              768            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1024            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1200            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1400            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1500            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1600            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1700            8              7                   0                      1              87.5                             0.0                               12.5
                   5.0             1725            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             1750            8              3                   0                      5              37.5                             0.0                               62.5
                   5.0             1800            8              5                   0                      3              62.5                             0.0                               37.5
                   5.0             2048            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             4096            8              8                   0                      0             100.0                             0.0                                0.0
                   5.0             8192            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0              512            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0              768            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1024            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1200            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1400            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1500            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1600            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             1700            8              7                   0                      1              87.5                             0.0                               12.5
                  10.0             1725            8              7                   0                      1              87.5                             0.0                               12.5
                  10.0             1750            8              2                   0                      6              25.0                             0.0                               75.0
                  10.0             1800            8              2                   0                      6              25.0                             0.0                               75.0
                  10.0             2048            8              7                   0                      1              87.5                             0.0                               12.5
                  10.0             4096            8              8                   0                      0             100.0                             0.0                                0.0
                  10.0             8192            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0              512            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0              768            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1024            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1200            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1400            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1500            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1600            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             1700            8              7                   0                      1              87.5                             0.0                               12.5
                  15.0             1725            8              7                   0                      1              87.5                             0.0                               12.5
                  15.0             1750            8              2                   0                      6              25.0                             0.0                               75.0
                  15.0             1800            8              1                   0                      7              12.5                             0.0                               87.5
                  15.0             2048            8              7                   0                      1              87.5                             0.0                               12.5
                  15.0             4096            8              8                   0                      0             100.0                             0.0                                0.0
                  15.0             8192            8              8                   0                      0             100.0                             0.0                                0.0

## Summary by Optimizer

 safety_margin_percent optimizer_name  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
                   0.0          adamw           98             89                   6                      3         90.816327                        6.122449                           3.061224
                   0.0            sgd           14             13                   1                      0         92.857143                        7.142857                           0.000000
                   5.0          adamw           98             89                   0                      9         90.816327                        0.000000                           9.183673
                   5.0            sgd           14             14                   0                      0        100.000000                        0.000000                           0.000000
                  10.0          adamw           98             83                   0                     15         84.693878                        0.000000                          15.306122
                  10.0            sgd           14             14                   0                      0        100.000000                        0.000000                           0.000000
                  15.0          adamw           98             82                   0                     16         83.673469                        0.000000                          16.326531
                  15.0            sgd           14             14                   0                      0        100.000000                        0.000000                           0.000000

## Remaining Dangerous Failures

model_name  batch_size  input_tokens optimizer_name                      experiment_type  safety_margin  safety_margin_percent  memory_limit_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  safe_predicted_reserved_MB  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_fits  predicted_fits  correct_prediction  dangerous_failure  conservative_failure                   outcome
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation            0.0                    0.0             1700                   1738.0                 1677.224945                 1677.224945                   1600.69                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation            0.0                    0.0             1725                   1738.0                 1677.224945                 1677.224945                   1600.69                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation            0.0                    0.0             1700                   1738.0                 1677.224945                 1677.224945                   1600.19                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation            0.0                    0.0             1725                   1738.0                 1677.224945                 1677.224945                   1600.19                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64            sgd      distilgpt2_optimizer_comparison            0.0                    0.0             1024                   1030.0                 1002.286117                 1002.286117                    956.49                   928.042701        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison            0.0                    0.0             1700                   1738.0                 1677.224945                 1677.224945                   1600.94                  1552.986061        False            True               False               True                 False dangerous_underprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison            0.0                    0.0             1725                   1738.0                 1677.224945                 1677.224945                   1600.94                  1552.986061        False            True               False               True                 False dangerous_underprediction

## Conservative Failures

model_name  batch_size  input_tokens optimizer_name                      experiment_type  safety_margin  safety_margin_percent  memory_limit_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  safe_predicted_reserved_MB  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_fits  predicted_fits  correct_prediction  dangerous_failure  conservative_failure                     outcome
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.05                    5.0             1700                   1700.0                 1634.028860                 1715.730303                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1700                   1700.0                 1634.028860                 1797.431746                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1725                   1700.0                 1634.028860                 1797.431746                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1750                   1700.0                 1634.028860                 1797.431746                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1700                   1700.0                 1634.028860                 1879.133189                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1725                   1700.0                 1634.028860                 1879.133189                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1750                   1700.0                 1634.028860                 1879.133189                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1800                   1700.0                 1634.028860                 1879.133189                   1592.43                  1512.989686         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation           0.05                    5.0             1750                   1738.0                 1677.224945                 1761.086193                   1600.69                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1750                   1738.0                 1677.224945                 1844.947440                   1600.69                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1800                   1738.0                 1677.224945                 1844.947440                   1600.69                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1750                   1738.0                 1677.224945                 1928.808687                   1600.69                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1800                   1738.0                 1677.224945                 1928.808687                   1600.69                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.00                    0.0             1750                   1726.0                 1763.617115                 1763.617115                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.05                    5.0             1750                   1726.0                 1763.617115                 1851.797971                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.05                    5.0             1800                   1726.0                 1763.617115                 1851.797971                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1750                   1726.0                 1763.617115                 1939.978827                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.10                   10.0             1800                   1726.0                 1763.617115                 1939.978827                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1750                   1726.0                 1763.617115                 2028.159683                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation           0.15                   15.0             1800                   1726.0                 1763.617115                 2028.159683                   1616.71                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation           0.05                    5.0             1750                   1738.0                 1677.224945                 1761.086193                   1600.19                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             1750                   1738.0                 1677.224945                 1844.947440                   1600.19                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             1800                   1738.0                 1677.224945                 1844.947440                   1600.19                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             1750                   1738.0                 1677.224945                 1928.808687                   1600.19                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             1800                   1738.0                 1677.224945                 1928.808687                   1600.19                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.00                    0.0             1750                   1726.0                 1763.617115                 1763.617115                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.05                    5.0             1750                   1726.0                 1763.617115                 1851.797971                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.05                    5.0             1800                   1726.0                 1763.617115                 1851.797971                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             1750                   1726.0                 1763.617115                 1939.978827                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             1800                   1726.0                 1763.617115                 1939.978827                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             1750                   1726.0                 1763.617115                 2028.159683                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             1800                   1726.0                 1763.617115                 2028.159683                   1616.21                  1632.978811         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.00                    0.0             1800                   1768.0                 1936.401455                 1936.401455                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.05                    5.0             1800                   1768.0                 1936.401455                 2033.221528                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             1800                   1768.0                 1936.401455                 2130.041601                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.10                   10.0             2048                   1768.0                 1936.401455                 2130.041601                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             1800                   1768.0                 1936.401455                 2226.861674                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation           0.15                   15.0             2048                   1768.0                 1936.401455                 2226.861674                   1648.43                  1792.964311         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison           0.05                    5.0             1750                   1738.0                 1677.224945                 1761.086193                   1600.94                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison           0.10                   10.0             1750                   1738.0                 1677.224945                 1844.947440                   1600.94                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison           0.10                   10.0             1800                   1738.0                 1677.224945                 1844.947440                   1600.94                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison           0.15                   15.0             1750                   1738.0                 1677.224945                 1928.808687                   1600.94                  1552.986061         True           False               False              False                  True conservative_overprediction
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison           0.15                   15.0             1800                   1738.0                 1677.224945                 1928.808687                   1600.94                  1552.986061         True           False               False              False                  True conservative_overprediction

## Realistic GPU Limit Summary

 safety_margin_percent  memory_limit_MB  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent
                   0.0             2048            8              8                   0                      0             100.0
                   0.0             4096            8              8                   0                      0             100.0
                   0.0             8192            8              8                   0                      0             100.0
                   5.0             2048            8              8                   0                      0             100.0
                   5.0             4096            8              8                   0                      0             100.0
                   5.0             8192            8              8                   0                      0             100.0
                  10.0             2048            8              7                   0                      1              87.5
                  10.0             4096            8              8                   0                      0             100.0
                  10.0             8192            8              8                   0                      0             100.0
                  15.0             2048            8              7                   0                      1              87.5
                  15.0             4096            8              8                   0                      0             100.0
                  15.0             8192            8              8                   0                      0             100.0

## Main Interpretation

Adding a safety margin should reduce dangerous underprediction.

The tradeoff is that conservative failures may increase, because the estimator becomes more cautious.

For scheduling, dangerous failures are worse than conservative failures because dangerous failures can cause OOM.

## Next Step

Use the selected safety margin in the training phase report and continue with model validation.

