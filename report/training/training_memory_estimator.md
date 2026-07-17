
# Day 27 - TrainingMemoryEstimator

## Goal

Build the first full training memory estimator.

The estimator combines:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory approximation
- framework overhead
- allocator padding

## File Created

- src/estimators/training_memory_estimator.py

## Formula

predicted_peak_allocated =
parameter_memory
+ gradient_memory
+ optimizer_state_memory
+ activation_memory
+ framework_overhead

predicted_peak_reserved =
predicted_peak_allocated
+ allocator_padding

## Estimator Settings

- activation_factor = 0.004
- framework_overhead_mb = 20.0
- allocator_padding_ratio = 0.08
- allocator_min_padding_mb = 32.0

## Prediction Results

model_name  batch_size  input_tokens dtype optimizer_name                      experiment_type  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  parameter_memory_MB  gradient_memory_MB  optimizer_state_memory_MB  activation_memory_MB  framework_overhead_MB  allocator_padding_MB  activation_factor  allocator_padding_ratio  allocated_absolute_error_MB  allocated_relative_error_percent  reserved_absolute_error_MB  reserved_relative_error_percent
distilgpt2           1            32  fp32          adamw distilgpt2_sequence_length_variation                   1592.43                  1309.883094                   1700.0                 1414.673741            312.47168           312.47168                 624.943359             39.996375                   20.0            104.790647              0.004                     0.08                   282.546906                         17.743129                  285.326259                        16.783898
distilgpt2           1            64  fp32          adamw distilgpt2_sequence_length_variation                   1600.69                  1349.879469                   1738.0                 1457.869826            312.47168           312.47168                 624.943359             79.992750                   20.0            107.990357              0.004                     0.08                   250.810531                         15.668901                  280.130174                        16.117962
distilgpt2           1           128  fp32          adamw distilgpt2_sequence_length_variation                   1616.71                  1429.872219                   1726.0                 1544.261996            312.47168           312.47168                 624.943359            159.985500                   20.0            114.389778              0.004                     0.08                   186.837781                         11.556666                  181.738004                        10.529432
distilgpt2           1            64  fp32          adamw      distilgpt2_batch_size_variation                   1600.19                  1349.879469                   1738.0                 1457.869826            312.47168           312.47168                 624.943359             79.992750                   20.0            107.990357              0.004                     0.08                   250.310531                         15.642551                  280.130174                        16.117962
distilgpt2           2            64  fp32          adamw      distilgpt2_batch_size_variation                   1616.21                  1429.872219                   1726.0                 1544.261996            312.47168           312.47168                 624.943359            159.985500                   20.0            114.389778              0.004                     0.08                   186.337781                         11.529305                  181.738004                        10.529432
distilgpt2           4            64  fp32          adamw      distilgpt2_batch_size_variation                   1648.43                  1589.857719                   1768.0                 1717.046336            312.47168           312.47168                 624.943359            319.971000                   20.0            127.188618              0.004                     0.08                    58.572281                          3.553216                   50.953664                         2.881995
distilgpt2           1            64  fp32            sgd      distilgpt2_optimizer_comparison                    956.49                   724.936109                   1030.0                  782.930998            312.47168           312.47168                   0.000000             79.992750                   20.0             57.994889              0.004                     0.08                   231.553891                         24.208710                  247.069002                        23.987282
distilgpt2           1            64  fp32          adamw      distilgpt2_optimizer_comparison                   1600.94                  1349.879469                   1738.0                 1457.869826            312.47168           312.47168                 624.943359             79.992750                   20.0            107.990357              0.004                     0.08                   251.060531                         15.682070                  280.130174                        16.117962

## Global Metrics

 num_rows  allocated_MRE  allocated_mean_error  allocated_max_error  allocated_min_error  reserved_MRE  reserved_mean_error  reserved_max_error  reserved_min_error
        8      15.655726             14.448069             24.20871             3.553216     16.117962             14.13324           23.987282            2.881995

## Error by Optimizer

optimizer_name  num_rows  allocated_mean_error  allocated_median_error  allocated_max_error  reserved_mean_error  reserved_median_error  reserved_max_error  avg_actual_allocated_MB  avg_predicted_allocated_MB  avg_actual_reserved_MB  avg_predicted_reserved_MB
         adamw         7             13.053691               15.642551            17.743129            12.725520              16.117962           16.783898                  1610.80                 1401.303379             1733.428571                1513.407650
           sgd         1             24.208710               24.208710            24.208710            23.987282              23.987282           23.987282                   956.49                  724.936109             1030.000000                 782.930998

## Error by Experiment

                     experiment_type  num_rows  allocated_mean_error  allocated_median_error  allocated_max_error  reserved_mean_error  reserved_median_error  reserved_max_error  avg_actual_allocated_MB  avg_predicted_allocated_MB  avg_actual_reserved_MB  avg_predicted_reserved_MB
     distilgpt2_batch_size_variation         3             10.241691               11.529305            15.642551             9.843130              10.529432           16.117962              1621.610000                 1456.536469             1744.000000                1573.059386
     distilgpt2_optimizer_comparison         2             19.945390               19.945390            24.208710            20.052622              20.052622           23.987282              1278.715000                 1037.407789             1384.000000                1120.400412
distilgpt2_sequence_length_variation         3             14.989565               15.668901            17.743129            14.477097              16.117962           16.783898              1603.276667                 1363.211594             1721.333333                1472.268521

## Component Summary

optimizer_name  parameter_memory_MB  gradient_memory_MB  optimizer_state_memory_MB  activation_memory_MB  framework_overhead_MB
         adamw            312.47168           312.47168                 624.943359            131.416661                   20.0
           sgd            312.47168           312.47168                   0.000000             79.992750                   20.0

## Main Interpretation

This is the first version of the TrainingMemoryEstimator.

The goal is not perfect accuracy yet.

The goal is to create a modular estimator that includes the major training memory components.

The OptimizerStateEstimator is now included inside the training estimator.

The next step is to evaluate the errors and improve the estimator formula.

