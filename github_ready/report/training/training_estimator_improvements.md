
# Day 28 - TrainingMemoryEstimator Improvement

## Goal

Improve the first TrainingMemoryEstimator.

Day 27 showed that the estimator underpredicted training memory.

Day 28 adds a backward temporary memory correction term.

## Main Change

Added:

backward_temp_memory_MB = parameter_memory_MB x backward_temp_factor

with:

backward_temp_factor = 0.65

## V1 vs V2 Metrics

                estimator  num_rows  allocated_MRE  allocated_mean_error  allocated_max_error  allocated_min_error  reserved_MRE  reserved_mean_error  reserved_max_error  reserved_min_error
TrainingMemoryEstimatorV1         8      15.655726             14.448069            24.208710             3.553216     16.117962            14.133240           23.987282            2.881995
TrainingMemoryEstimatorV2         8       2.977173              3.462507             8.767998             1.006291      3.496839             3.868211            9.524969            2.179439

## V2 Predictions

model_name  batch_size  input_tokens dtype optimizer_name                      experiment_type  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  parameter_memory_MB  gradient_memory_MB  optimizer_state_memory_MB  activation_memory_MB  backward_temp_memory_MB  framework_overhead_MB  allocator_padding_MB  activation_factor  backward_temp_factor  allocator_padding_ratio  allocated_absolute_error_MB  allocated_relative_error_percent  reserved_absolute_error_MB  reserved_relative_error_percent
distilgpt2           1            32  fp32          adamw distilgpt2_sequence_length_variation                   1592.43                  1512.989686                   1700.0                 1634.028860            312.47168           312.47168                 624.943359             39.996375               203.106592                   20.0            121.039175              0.004                  0.65                     0.08                    79.440314                          4.988622                   65.971140                         3.880655
distilgpt2           1            64  fp32          adamw distilgpt2_sequence_length_variation                   1600.69                  1552.986061                   1738.0                 1677.224945            312.47168           312.47168                 624.943359             79.992750               203.106592                   20.0            124.238885              0.004                  0.65                     0.08                    47.703939                          2.980211                   60.775055                         3.496839
distilgpt2           1           128  fp32          adamw distilgpt2_sequence_length_variation                   1616.71                  1632.978811                   1726.0                 1763.617115            312.47168           312.47168                 624.943359            159.985500               203.106592                   20.0            130.638305              0.004                  0.65                     0.08                    16.268811                          1.006291                   37.617115                         2.179439
distilgpt2           1            64  fp32          adamw      distilgpt2_batch_size_variation                   1600.19                  1552.986061                   1738.0                 1677.224945            312.47168           312.47168                 624.943359             79.992750               203.106592                   20.0            124.238885              0.004                  0.65                     0.08                    47.203939                          2.949896                   60.775055                         3.496839
distilgpt2           2            64  fp32          adamw      distilgpt2_batch_size_variation                   1616.21                  1632.978811                   1726.0                 1763.617115            312.47168           312.47168                 624.943359            159.985500               203.106592                   20.0            130.638305              0.004                  0.65                     0.08                    16.768811                          1.037539                   37.617115                         2.179439
distilgpt2           4            64  fp32          adamw      distilgpt2_batch_size_variation                   1648.43                  1792.964311                   1768.0                 1936.401455            312.47168           312.47168                 624.943359            319.971000               203.106592                   20.0            143.437145              0.004                  0.65                     0.08                   144.534311                          8.767998                  168.401455                         9.524969
distilgpt2           1            64  fp32            sgd      distilgpt2_optimizer_comparison                    956.49                   928.042701                   1030.0                 1002.286117            312.47168           312.47168                   0.000000             79.992750               203.106592                   20.0             74.243416              0.004                  0.65                     0.08                    28.447299                          2.974134                   27.713883                         2.690668
distilgpt2           1            64  fp32          adamw      distilgpt2_optimizer_comparison                   1600.94                  1552.986061                   1738.0                 1677.224945            312.47168           312.47168                 624.943359             79.992750               203.106592                   20.0            124.238885              0.004                  0.65                     0.08                    47.953939                          2.995361                   60.775055                         3.496839

## Row-wise V1 vs V2 Comparison

model_name  batch_size  input_tokens optimizer_name                      experiment_type  actual_peak_allocated_MB  v1_predicted_allocated_MB  v1_allocated_error_percent  actual_peak_reserved_MB  v1_predicted_reserved_MB  v1_reserved_error_percent  v2_predicted_allocated_MB  v2_allocated_error_percent  v2_predicted_reserved_MB  v2_reserved_error_percent  allocated_error_improvement_percent_points  reserved_error_improvement_percent_points
distilgpt2           1            32          adamw distilgpt2_sequence_length_variation                   1592.43                1309.883094                   17.743129                   1700.0               1414.673741                  16.783898                1512.989686                    4.988622               1634.028860                   3.880655                                   12.754507                                  12.903242
distilgpt2           1            64          adamw distilgpt2_sequence_length_variation                   1600.69                1349.879469                   15.668901                   1738.0               1457.869826                  16.117962                1552.986061                    2.980211               1677.224945                   3.496839                                   12.688690                                  12.621123
distilgpt2           1           128          adamw distilgpt2_sequence_length_variation                   1616.71                1429.872219                   11.556666                   1726.0               1544.261996                  10.529432                1632.978811                    1.006291               1763.617115                   2.179439                                   10.550375                                   8.349994
distilgpt2           1            64          adamw      distilgpt2_batch_size_variation                   1600.19                1349.879469                   15.642551                   1738.0               1457.869826                  16.117962                1552.986061                    2.949896               1677.224945                   3.496839                                   12.692655                                  12.621123
distilgpt2           2            64          adamw      distilgpt2_batch_size_variation                   1616.21                1429.872219                   11.529305                   1726.0               1544.261996                  10.529432                1632.978811                    1.037539               1763.617115                   2.179439                                   10.491766                                   8.349994
distilgpt2           4            64          adamw      distilgpt2_batch_size_variation                   1648.43                1589.857719                    3.553216                   1768.0               1717.046336                   2.881995                1792.964311                    8.767998               1936.401455                   9.524969                                   -5.214782                                  -6.642975
distilgpt2           1            64            sgd      distilgpt2_optimizer_comparison                    956.49                 724.936109                   24.208710                   1030.0                782.930998                  23.987282                 928.042701                    2.974134               1002.286117                   2.690668                                   21.234576                                  21.296614
distilgpt2           1            64          adamw      distilgpt2_optimizer_comparison                   1600.94                1349.879469                   15.682070                   1738.0               1457.869826                  16.117962                1552.986061                    2.995361               1677.224945                   3.496839                                   12.686709                                  12.621123

## V2 Error by Optimizer

optimizer_name  num_rows  allocated_mean_error  allocated_median_error  allocated_max_error  reserved_mean_error  reserved_median_error  reserved_max_error  avg_actual_allocated_MB  avg_predicted_allocated_MB  avg_actual_reserved_MB  avg_predicted_reserved_MB
         adamw         7              3.532274                2.980211             8.767998             4.036431               3.496839            9.524969                  1610.80                 1604.409971             1733.428571                1732.762769
           sgd         1              2.974134                2.974134             2.974134             2.690668               2.690668            2.690668                   956.49                  928.042701             1030.000000                1002.286117

## V2 Error by Experiment

                     experiment_type  num_rows  allocated_mean_error  allocated_median_error  allocated_max_error  reserved_mean_error  reserved_median_error  reserved_max_error  avg_actual_allocated_MB  avg_predicted_allocated_MB  avg_actual_reserved_MB  avg_predicted_reserved_MB
     distilgpt2_batch_size_variation         3              4.251811                2.949896             8.767998             5.067082               3.496839            9.524969              1621.610000                 1659.643061             1744.000000                1792.414505
     distilgpt2_optimizer_comparison         2              2.984748                2.984748             2.995361             3.093753               3.093753            3.496839              1278.715000                 1240.514381             1384.000000                1339.755531
distilgpt2_sequence_length_variation         3              2.991708                2.980211             4.988622             3.185644               3.496839            3.880655              1603.276667                 1566.318186             1721.333333                1691.623640

## V2 Component Summary

optimizer_name  parameter_memory_MB  gradient_memory_MB  optimizer_state_memory_MB  activation_memory_MB  backward_temp_memory_MB  framework_overhead_MB
         adamw            312.47168           312.47168                 624.943359            131.416661               203.106592                   20.0
           sgd            312.47168           312.47168                   0.000000             79.992750               203.106592                   20.0

## Interpretation

V1 underpredicted training memory because it did not include backward temporary memory.

V2 adds a backward temporary correction based on parameter memory.

This is still an explainable estimator, not a black-box fit.

The goal is to reduce systematic underprediction while keeping the formula simple.

## Next Step

Use the improved estimator for training PEF-style evaluation and final training reports.

