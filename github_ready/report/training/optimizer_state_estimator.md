
# Day 26 - OptimizerStateEstimator

## Goal

Build an estimator for optimizer-state memory.

Day 25 showed that AdamW uses much more memory than SGD during optimizer_step.

Day 26 converts this observation into a reusable estimator module.

## File Created

- src/estimators/optimizer_state_estimator.py

## Estimator Rule

The estimator uses optimizer-specific state factors:

- SGD: 0 x parameter memory
- SGD with momentum: 1 x parameter memory
- Adam: 2 x parameter memory
- AdamW: 2 x parameter memory

## Optimizer State Predictions

model_name optimizer_name dtype  actual_peak_allocated_MB  actual_peak_reserved_MB  final_allocated_MB  final_reserved_MB  parameter_memory_MB  optimizer_state_factor  estimated_optimizer_state_MB
distilgpt2            sgd  fp32                    956.49                   1030.0              345.63             1030.0            312.47168                     0.0                      0.000000
distilgpt2          adamw  fp32                   1600.94                   1738.0              973.48             1738.0            312.47168                     2.0                    624.943359

## Difference Summary

model_name  observed_peak_allocated_difference_MB  observed_peak_reserved_difference_MB  estimated_optimizer_state_difference_MB  absolute_error_vs_peak_allocated_difference_MB  relative_error_vs_peak_allocated_difference_percent
distilgpt2                                 644.45                                 708.0                               624.943359                                       19.506641                                             3.026866

## Optimizer Step Jump Comparison

model_name optimizer_name  estimated_optimizer_state_MB  actual_allocated_jump_backward_to_step_MB  actual_peak_allocated_jump_backward_to_step_MB  actual_reserved_jump_backward_to_step_MB  error_vs_current_allocated_jump_MB  error_vs_peak_allocated_jump_MB
distilgpt2          adamw                    624.943359                                     627.22                                          644.45                                     688.0                            2.276641                        19.506641

## Metrics

model_name optimizer  estimated_optimizer_state_MB  observed_peak_allocated_difference_MB  observed_peak_reserved_difference_MB  absolute_error_MB  relative_error_percent
distilgpt2     adamw                    624.943359                                 644.45                                 708.0          19.506641                3.026866

## Interpretation

AdamW stores first and second moment tensors, so its optimizer state is approximated as 2 times parameter memory.

SGD without momentum has little persistent optimizer state, so its optimizer-state factor is modeled as zero.

The Day 25 experiment supports this because AdamW had a large memory jump after optimizer_step, while SGD did not.

## Next Step

Use this OptimizerStateEstimator inside the TrainingMemoryEstimator.

