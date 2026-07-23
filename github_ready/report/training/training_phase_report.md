# Training Phase Report

## 1. Overview

This report summarizes the training-memory phase of the xMem-inspired LLM GPU memory profiling and prediction project.

The earlier phase focused on inference memory profiling and prediction. This phase extends the project to training memory, where peak memory depends on parameters, gradients, activations, optimizer states, backward temporary tensors, and CUDA allocator behavior.

The training phase covers:

- training memory logging
- sequence-length variation
- batch-size variation
- optimizer comparison
- optimizer-state estimation
- full training-memory estimation
- PEF-style fit/fail evaluation
- safety-margin analysis

## 2. Training Logger

A training memory logger was built to capture CUDA memory at key training stages:

- before_model_load
- after_model_load
- after_batch_creation
- after_optimizer_creation
- after_forward
- after_loss
- after_backward
- after_optimizer_step
- after_zero_grad

For each stage, the logger records allocated memory, reserved memory, peak allocated memory, and peak reserved memory.

This stage-level logging is important because peak training memory can occur temporarily during backward or optimizer step.

## 3. tiny-gpt2 Training Sanity Check

The logger was first tested on `sshleifer/tiny-gpt2`.

For tiny-gpt2, peak allocated memory increased from 27.27 MB at 16 tokens to 118.95 MB at 128 tokens.

This confirmed that the logger worked correctly. However, tiny-gpt2 is too small for final conclusions.

## 4. distilgpt2 Sequence-length Training Experiment

The first meaningful training experiment used `distilgpt2` with batch_size=1, dtype=fp32, optimizer=AdamW, and input_tokens = 32, 64, 128.

| input_tokens | peak_allocated_MB | peak_reserved_MB |
|---:|---:|---:|
| 32 | 1592.43 | 1700.0 |
| 64 | 1600.69 | 1738.0 |
| 128 | 1616.71 | 1726.0 |

Peak allocated memory increased from 1592.43 MB to 1616.71 MB.

The increase was real but not very large because training memory is dominated by fixed components such as parameters, gradients, and optimizer states.

## 5. Training vs Inference

For `distilgpt2`, training used around 4.7x to 4.8x more peak allocated memory than inference under comparable token settings.

This confirmed that training memory requires a separate estimator and cannot be handled by the inference estimator alone.

## 6. distilgpt2 Batch-size Training Experiment

The batch-size experiment used model=distilgpt2, input_tokens=64, dtype=fp32, optimizer=AdamW, and batch_size = 1, 2, 4.

| batch_size | peak_allocated_MB | peak_reserved_MB |
|---:|---:|---:|
| 1 | 1600.19 | 1738.0 |
| 2 | 1616.21 | 1726.0 |
| 4 | 1648.43 | 1768.0 |

Peak allocated memory increased from 1600.19 MB to 1648.43 MB.

The scaling was sublinear because model parameters, gradients, and optimizer states are shared across the batch.

## 7. Optimizer Comparison: SGD vs AdamW

The optimizer comparison used model=distilgpt2, batch_size=1, input_tokens=64, dtype=fp32, and optimizers SGD and AdamW.

| optimizer | peak_allocated_MB | peak_reserved_MB | final_allocated_MB |
|---|---:|---:|---:|
| SGD | 956.49 | 1030.0 | 345.63 |
| AdamW | 1600.94 | 1738.0 | 973.48 |

AdamW used 644.45 MB more peak allocated memory than SGD.

The stage-wise logs showed that both optimizers had the same peak after backward, but AdamW jumped sharply after optimizer_step.

This showed that AdamW optimizer state is a major contributor to training memory.

## 8. OptimizerStateEstimator

An OptimizerStateEstimator was built using optimizer-specific memory factors:

| optimizer | optimizer_state_factor |
|---|---:|
| SGD | 0x parameter memory |
| SGD with momentum | 1x parameter memory |
| Adam | 2x parameter memory |
| AdamW | 2x parameter memory |

For distilgpt2 fp32:

- parameter memory = 312.47 MB
- AdamW estimated optimizer-state memory = 624.94 MB
- observed AdamW-SGD peak allocated difference = 644.45 MB

The relative error was 3.03%.

This supports the rule that AdamW optimizer state is approximately 2x parameter memory.

## 9. TrainingMemoryEstimator V1

The first training estimator used:

predicted_peak_allocated = parameter_memory + gradient_memory + optimizer_state_memory + activation_memory + framework_overhead

V1 result:

| metric | value |
|---|---:|
| allocated MRE | 15.66% |
| allocated mean error | 14.45% |
| reserved MRE | 16.12% |
| reserved mean error | 14.13% |

V1 systematically underpredicted memory, especially for SGD.

This showed that another component was missing.

## 10. TrainingMemoryEstimator V2

V2 added backward temporary memory:

backward_temp_memory_MB = parameter_memory_MB x 0.65

For distilgpt2, this added 203.11 MB.

V2 result:

| metric | value |
|---|---:|
| allocated MRE | 2.98% |
| allocated mean error | 3.46% |
| reserved MRE | 3.50% |
| reserved mean error | 3.87% |

This was a major improvement over V1.

The improvement showed that backward temporary memory is an important training-memory component.

## 11. V1 vs V2 Comparison

| estimator | allocated_MRE | reserved_MRE |
|---|---:|---:|
| V1 | 15.66% | 16.12% |
| V2 | 2.98% | 3.50% |

V2 is much stronger.

However, V2 slightly overpredicted the batch_size=4 AdamW row. This is acceptable because the overall error improved strongly.

## 12. Training PEF-style Fit/Fail Simulation

A PEF-style simulation was run using actual and predicted reserved memory.

Without safety margin:

| metric | value |
|---|---:|
| total cases | 112 |
| accuracy | 91.07% |
| dangerous failures | 7 |
| conservative failures | 3 |

Dangerous failures occurred near tight boundary limits, especially around 1700 MB and 1725 MB.

## 13. Safety-margin Evaluation

A safety margin was added to predicted reserved memory:

safe_predicted_reserved_MB = predicted_peak_reserved_MB x (1 + safety_margin)

Margins tested: 0%, 5%, 10%, 15%.

| margin | accuracy | dangerous failures | conservative failures |
|---:|---:|---:|---:|
| 0% | 91.07% | 7 | 3 |
| 5% | 91.96% | 0 | 9 |
| 10% | 86.61% | 0 | 15 |
| 15% | 85.71% | 0 | 16 |

The 5% margin was the best choice.

It removed all dangerous failures and slightly improved accuracy.

## 14. Key Findings

| finding_id | finding | evidence | importance |
|---:|---|---|---|
| 1 | Training memory is much higher than inference memory for distilgpt2. | Training used around 4.7x to 4.8x more peak allocated memory than inference under comparable settings. | Training memory needs separate modeling from inference memory. |
| 2 | Training memory increases with sequence length, but fixed training components dominate. | Peak allocated memory increased from 1592.43 MB at 32 tokens to 1616.71 MB at 128 tokens. | Sequence length matters, but parameters, gradients, and optimizer states dominate in this setup. |
| 3 | Batch-size scaling is sublinear. | Peak allocated memory increased from 1600.19 MB at batch size 1 to 1648.43 MB at batch size 4. | Model weights, gradients, and optimizer states are shared across the batch. |
| 4 | AdamW optimizer state is a major contributor to training memory. | AdamW used 1600.94 MB peak allocated memory while SGD used 956.49 MB. | Optimizer choice must be included in training memory prediction. |
| 5 | AdamW optimizer state can be approximated as 2x parameter memory. | Estimated optimizer-state memory was 624.94 MB; observed AdamW-SGD peak difference was 644.45 MB. | Supports the OptimizerStateEstimator design. |
| 6 | Backward temporary memory correction significantly improves training prediction. | Allocated MRE improved from 15.66% in V1 to 2.98% in V2. | Backward temporary memory is a required component for training prediction. |
| 7 | A 5% safety margin removes dangerous fit/fail failures. | Dangerous failures reduced from 7 to 0, while accuracy improved from 91.07% to 91.96%. | Makes the estimator safer for scheduling-style decisions. |

## 15. Key Metrics

| section | metric | value | source |
|---|---|---|---|
| training_vs_inference | distilgpt2_training_memory_ratio | around 4.7x to 4.8x inference memory | training sequence experiment |
| sequence_length_training | peak_allocated_32_to_128_tokens | 1592.43 MB to 1616.71 MB | distilgpt2 sequence-length training |
| batch_size_training | peak_allocated_batch_1_to_4 | 1600.19 MB to 1648.43 MB | distilgpt2 batch-size training |
| optimizer_comparison | adamw_vs_sgd_peak_allocated_difference | 644.45 MB | SGD vs AdamW training comparison |
| optimizer_state_estimator | adamw_optimizer_state_relative_error | 3.03% | optimizer-state estimator evaluation |
| training_estimator | training_estimator_v2_allocated_MRE | 2.98% | training estimator V2 evaluation |
| training_estimator | training_estimator_v2_reserved_MRE | 3.50% | training estimator V2 evaluation |
| training_pef | fit_fail_accuracy_without_margin | 91.07% | training PEF simulation |
| training_pef_safety_margin | best_safety_margin | 5% | safety-margin PEF simulation |
| training_pef_safety_margin | dangerous_failures_after_5_percent_margin | 0 | safety-margin PEF simulation |

## 16. Limitations

- Experiments are limited to single-GPU Colab/T4-style runs.
- The main training model is distilgpt2.
- Larger models such as gpt2 still need validation.
- The backward temporary correction is empirical.
- The safety margin is evaluated on a small dataset.
- Real multi-GPU model parallelism is not implemented.
- Sparsity and quantization training experiments are not yet included.

## 17. Next Steps

- validate the estimators on gpt2
- run limited gpt2 inference experiments
- run limited gpt2 training experiments
- test whether optimizer-state and backward-temp formulas generalize
- update README and GitHub structure
- prepare final combined report after additional validation

## 18. Conclusion

The training phase successfully extended the project from inference memory prediction to training memory prediction.

The key outcome is that training memory can be modeled using parameter memory, gradient memory, optimizer-state memory, activation memory, backward temporary memory, allocator padding, and safety margin for fit/fail decisions.

TrainingMemoryEstimator V2 achieved 2.98% allocated MRE and 3.50% reserved MRE on the current distilgpt2 training dataset.

A 5% safety margin removed all dangerous fit/fail failures in the PEF-style simulation.

This makes the training estimator useful not only for numeric prediction, but also for scheduling-style fit/fail decisions.