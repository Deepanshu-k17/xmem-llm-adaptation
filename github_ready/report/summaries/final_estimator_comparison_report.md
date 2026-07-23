# Final Estimator Comparison Report

## 1. Goal

This report summarizes the major estimator results produced so far in the xMem-inspired LLM memory profiling and prediction project.

It compares inference estimators, training estimators, PEF-style fit/fail prediction, and safety-margin behavior.

This is not the final full project report. It is the final estimator comparison before moving to quantization, sparsity, model-parallelism, and architecture comparison.

## 2. Final Estimator Comparison Table

| phase     | estimator                                  | validation_scope           |   allocated_MRE_percent |   allocated_mean_error_percent |   allocated_max_error_percent |   reserved_MRE_percent |   reserved_mean_error_percent |   reserved_max_error_percent |   pef_accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |   safety_margin_percent | status                                                             |
|:----------|:-------------------------------------------|:---------------------------|------------------------:|-------------------------------:|------------------------------:|-----------------------:|------------------------------:|-----------------------------:|-----------------------:|---------------------------------:|------------------------------------:|------------------------:|:-------------------------------------------------------------------|
| inference | CombinedInferenceEstimator                 | distilgpt2                 |                  nan    |                           2.36 |                        nan    |                 nan    |                          2.36 |                       nan    |                  99.52 |                             0    |                              nan    |                     0   | strong inference result on distilgpt2                              |
| inference | CombinedInferenceEstimator                 | gpt2                       |                    2.66 |                           2.5  |                          5.09 |                   2.06 |                          2.28 |                         5.22 |                 nan    |                           nan    |                              nan    |                     0   | validated inference generalization to larger GPT-style model       |
| training  | TrainingMemoryEstimator V1                 | distilgpt2                 |                   15.66 |                          14.45 |                         24.21 |                  16.12 |                         14.13 |                        23.99 |                 nan    |                           nan    |                              nan    |                     0   | baseline; underpredicted due to missing backward temporary memory  |
| training  | TrainingMemoryEstimator V2                 | distilgpt2 + gpt2 combined |                    2.99 |                           4.95 |                         17.69 |                   3.69 |                          5    |                        16.83 |                 nan    |                           nan    |                              nan    |                     0   | good baseline training estimator but weak on gpt2 SGD              |
| training  | TrainingMemoryEstimator V3                 | distilgpt2 + gpt2 combined |                    2.99 |                           4.36 |                         12.77 |                   4.2  |                          4.99 |                        12.52 |                 nan    |                           nan    |                              nan    |                     0   | experimental optimizer-specific correction; damaged distilgpt2 SGD |
| training  | TrainingMemoryEstimator V4                 | distilgpt2 + gpt2 combined |                    2.98 |                           3.66 |                          8.77 |                   3.69 |                          4.29 |                         9.52 |                  96.24 |                             3.01 |                                0.75 |                     0   | candidate final training estimator for raw prediction              |
| training  | TrainingMemoryEstimator V4 + safety margin | distilgpt2 + gpt2 combined |                    2.98 |                           3.66 |                          8.77 |                   3.69 |                          4.29 |                         9.52 |                  95.11 |                             0    |                                4.89 |                     7.5 | recommended safe setting for training placement                    |

## 3. Phase Summary

| phase     | best_estimator_for_prediction   | best_validation_result                                                            | best_pef_result                                                 | main_memory_components                                                                             | main_conclusion                                                                                                    |
|:----------|:--------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| inference | CombinedInferenceEstimator      | gpt2 inference: 2.50% allocated mean error, 2.28% reserved mean error             | distilgpt2 inference: 99.52% PEF accuracy, 0 dangerous failures | parameters, activations, KV cache, dtype, allocator behavior                                       | Inference memory can be predicted accurately using modular parameter/token/cache/precision/allocator corrections.  |
| training  | TrainingMemoryEstimator V4      | combined distilgpt2 + gpt2: 3.66% allocated mean error, 4.29% reserved mean error | V4 + 7.5% safety margin: 95.11% accuracy, 0 dangerous failures  | parameters, gradients, optimizer states, activations, backward temporary memory, allocator padding | Training memory needs separate modeling because optimizer state and backward temporary memory dominate peak usage. |

## 4. Recommended Estimators

| use_case                           | recommended_estimator                           | reason                                                                   | important_metric                                               | caution                                                                     |
|:-----------------------------------|:------------------------------------------------|:-------------------------------------------------------------------------|:---------------------------------------------------------------|:----------------------------------------------------------------------------|
| LLM inference memory prediction    | CombinedInferenceEstimator                      | Strong low-error prediction on distilgpt2 and gpt2 inference validation. | gpt2 allocated mean error = 2.50%, reserved mean error = 2.28% | Tiny-gpt2 is too small for realistic evaluation because overhead dominates. |
| LLM training raw memory prediction | TrainingMemoryEstimator V4                      | Best combined training estimator across distilgpt2 and gpt2.             | allocated mean error = 3.66%, reserved mean error = 4.29%      | 100M parameter threshold is empirical and needs more model validation.      |
| safe training workload placement   | TrainingMemoryEstimator V4 + 7.5% safety margin | Removes all dangerous underpredictions in current validation.            | 0 dangerous failures, 95.11% PEF accuracy                      | Increases conservative failures, so GPU utilization may decrease.           |

## 5. Key Findings

|   finding_id | finding                                                          | evidence                                                                                                       | why_it_matters                                                                                    |
|-------------:|:-----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
|            1 | Inference memory prediction generalizes from distilgpt2 to gpt2. | gpt2 inference achieved 2.50% allocated mean error and 2.28% reserved mean error.                              | Shows estimator is not only fitted to distilgpt2.                                                 |
|            2 | Training memory needs separate modeling from inference memory.   | Training used around 4.7x to 4.8x more memory than comparable inference runs.                                  | Gradients, optimizer states, and backward temporary memory make training fundamentally different. |
|            3 | AdamW optimizer state is close to 2x parameter memory.           | distilgpt2 estimate 624.94 MB vs observed 644.45 MB; gpt2 estimate 949.40 MB vs observed ~950.92 MB.           | Optimizer-state modeling is essential for training memory prediction.                             |
|            4 | V4 is the strongest current training estimator.                  | V4 achieved 3.66% allocated mean error, 4.29% reserved mean error, and reduced max errors compared with V2/V3. | V4 becomes the candidate final training estimator.                                                |
|            5 | PEF-style evaluation is stricter than MRE.                       | V4 had low error but still produced dangerous failures near tight memory limits.                               | Deployment decisions require fit/fail evaluation, not only numeric prediction error.              |
|            6 | V4 needs a 7.5% safety margin for safe training placement.       | 7.5% margin reduced dangerous failures to 0 with 95.11% accuracy.                                              | Safe scheduling prioritizes avoiding OOM over maximizing raw accuracy.                            |

## 6. Inference Interpretation

The inference estimator performs well because inference memory is dominated by relatively stable components such as model parameters, activation memory, KV cache, dtype, and allocator behavior.

The gpt2 inference validation supports generalization beyond distilgpt2, with 2.50% allocated mean error and 2.28% reserved mean error.

## 7. Training Interpretation

Training memory is more complex because it includes gradients, optimizer states, backward temporary tensors, and optimizer-step memory behavior.

TrainingMemoryEstimator V1 underpredicted because it missed backward temporary memory.

V2 added backward temporary correction and improved strongly.

V3 tested optimizer-specific correction but damaged distilgpt2 SGD.

V4 added model-size-aware correction and became the strongest current training estimator.

## 8. PEF and Safety Interpretation

PEF-style evaluation is stricter than MRE because it checks actual placement decisions under memory limits.

TrainingMemoryEstimator V4 achieved 96.24% PEF accuracy without margin but still had dangerous failures.

Adding a 7.5% safety margin removed all dangerous failures, reducing unsafe OOM-risk decisions to zero in the current validation set.

The tradeoff is more conservative failures, which can reduce GPU utilization.

## 9. Current Recommendations

- Use CombinedInferenceEstimator for inference memory prediction.
- Use TrainingMemoryEstimator V4 for raw training memory prediction.
- Use TrainingMemoryEstimator V4 + 7.5% safety margin for safe training workload placement.

## 10. Limitations

- Current validation covers distilgpt2 and gpt2 only.
- The V4 100M parameter threshold is empirical.
- Real model parallelism is not implemented yet.
- Quantization and sparsity modules are still pending.
- CNN/Vision Transformer comparison is still pending.
- Reserved memory may vary across PyTorch/CUDA runtime environments.

## 11. Next Step

The next phase will extend the project toward the official project scope by adding precision/quantization memory estimation, sparsity memory estimation, model-parallel partitioning simulation, and CNN/Transformer architecture comparison.