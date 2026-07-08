# ModelParallelMemoryEstimator Report

## 1. Goal

This report documents the reusable ModelParallelMemoryEstimator module added to the project.

The module converts the Day 52 model-parallel memory formulas into reusable code under `src/estimators/`.

## 2. Module Path

`src/estimators/model_parallel_memory_estimator.py`

## 3. Supported Dtypes

- fp32
- fp16
- bf16

## 4. What the Estimator Computes

The estimator computes:

- total parameter memory
- ideal partitioned parameter memory
- replicated overhead memory
- communication buffer memory
- total overhead memory
- estimated per-device memory
- ideal reduction percentage
- effective reduction percentage

## 5. Important Scope Clarification

This is an analytical estimator.

It does not run tensor parallelism, pipeline parallelism, NCCL communication, or distributed PyTorch.

It estimates per-device parameter-memory behavior based on parameter count, dtype, device count, and overhead assumptions.

## 6. Estimator Demo

| model_name   |   num_parameters | dtype   |   num_devices |   total_parameter_memory_MB |   ideal_partitioned_parameter_memory_MB |   replication_overhead_percent |   communication_buffer_percent |   replicated_overhead_memory_MB |   communication_buffer_memory_MB |   total_overhead_memory_MB |   estimated_per_device_memory_MB |   ideal_reduction_percent |   effective_reduction_percent |   overhead_fraction_of_total_percent | scope_note                                                               |
|:-------------|-----------------:|:--------|--------------:|----------------------------:|----------------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|---------------------------------:|---------------------------:|---------------------------------:|--------------------------:|------------------------------:|-------------------------------------:|:-------------------------------------------------------------------------|
| distilgpt2   |         81912576 | fp32    |             1 |                     312.472 |                                312.472  |                              5 |                              3 |                        15.6236  |                          9.37415 |                    24.9977 |                         337.469  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp32    |             2 |                     312.472 |                                156.236  |                              5 |                              3 |                        15.6236  |                          9.37415 |                    24.9977 |                         181.234  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp32    |             4 |                     312.472 |                                 78.1179 |                              5 |                              3 |                        15.6236  |                          9.37415 |                    24.9977 |                         103.116  |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp32    |             8 |                     312.472 |                                 39.059  |                              5 |                              3 |                        15.6236  |                          9.37415 |                    24.9977 |                          64.0567 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp16    |             1 |                     156.236 |                                156.236  |                              5 |                              3 |                         7.81179 |                          4.68708 |                    12.4989 |                         168.735  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp16    |             2 |                     156.236 |                                 78.1179 |                              5 |                              3 |                         7.81179 |                          4.68708 |                    12.4989 |                          90.6168 |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp16    |             4 |                     156.236 |                                 39.059  |                              5 |                              3 |                         7.81179 |                          4.68708 |                    12.4989 |                          51.5578 |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| distilgpt2   |         81912576 | fp16    |             8 |                     156.236 |                                 19.5295 |                              5 |                              3 |                         7.81179 |                          4.68708 |                    12.4989 |                          32.0283 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp32    |             1 |                     474.7   |                                474.7    |                              5 |                              3 |                        23.735   |                         14.241   |                    37.976  |                         512.676  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp32    |             2 |                     474.7   |                                237.35   |                              5 |                              3 |                        23.735   |                         14.241   |                    37.976  |                         275.326  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp32    |             4 |                     474.7   |                                118.675  |                              5 |                              3 |                        23.735   |                         14.241   |                    37.976  |                         156.651  |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp32    |             8 |                     474.7   |                                 59.3375 |                              5 |                              3 |                        23.735   |                         14.241   |                    37.976  |                          97.3135 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp16    |             1 |                     237.35  |                                237.35   |                              5 |                              3 |                        11.8675  |                          7.1205  |                    18.988  |                         256.338  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp16    |             2 |                     237.35  |                                118.675  |                              5 |                              3 |                        11.8675  |                          7.1205  |                    18.988  |                         137.663  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp16    |             4 |                     237.35  |                                 59.3375 |                              5 |                              3 |                        11.8675  |                          7.1205  |                    18.988  |                          78.3255 |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |
| gpt2         |        124439808 | fp16    |             8 |                     237.35  |                                 29.6688 |                              5 |                              3 |                        11.8675  |                          7.1205  |                    18.988  |                          48.6568 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real multi-GPU execution. |

## 7. Summary

| dtype   |   num_devices |   avg_estimated_per_device_memory_MB |   avg_ideal_reduction_percent |   avg_effective_reduction_percent |   avg_overhead_fraction_percent |
|:--------|--------------:|-------------------------------------:|------------------------------:|----------------------------------:|--------------------------------:|
| fp16    |             1 |                             212.536  |                           0   |                              -8   |                               8 |
| fp16    |             2 |                             114.14   |                          50   |                              42   |                               8 |
| fp16    |             4 |                              64.9417 |                          75   |                              67   |                               8 |
| fp16    |             8 |                              40.3426 |                          87.5 |                              79.5 |                               8 |
| fp32    |             1 |                             425.073  |                           0   |                              -8   |                               8 |
| fp32    |             2 |                             228.28   |                          50   |                              42   |                               8 |
| fp32    |             4 |                             129.883  |                          75   |                              67   |                               8 |
| fp32    |             8 |                              80.6851 |                          87.5 |                              79.5 |                               8 |

## 8. Key Findings

|   finding_id | finding                                                                          | evidence                                                                                                                    | why_it_matters                                                                      |
|-------------:|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
|            1 | ModelParallelMemoryEstimator reproduces Day 52 analytical partitioning behavior. | Effective fp32 reductions are 42.00% at 2 devices, 67.00% at 4 devices, and 79.50% at 8 devices.                            | Model-parallel memory logic is now reusable project code.                           |
|            2 | The estimator separates ideal partitioned memory from overhead memory.           | Each row includes ideal_partitioned_parameter_memory_MB, replicated_overhead_memory_MB, and communication_buffer_memory_MB. | This avoids pretending model-parallel scaling is perfectly linear.                  |
|            3 | Combined overhead reduces ideal scaling by 8 percentage points.                  | The estimator uses 5% replication overhead and 3% communication buffer overhead.                                            | Distributed memory planning must include communication and replicated-memory costs. |
|            4 | The module supports fp32, fp16, and bf16 parameter-memory estimation.            | supported_dtypes() returns fp32, fp16, and bf16.                                                                            | The estimator can be combined later with precision-aware memory planning.           |
|            5 | This is analytical model-parallel estimation, not real multi-GPU profiling.      | No tensor parallelism, pipeline parallelism, NCCL communication, or distributed runtime was executed.                       | The project should claim model-parallel simulation, not implementation.             |

## 9. Why This Module Matters

Before this module, model-parallel calculations existed only in notebook-level tables.

Now the project has a reusable model-parallel estimator that can be used by later reports and comparison modules.

## 10. Limitations

- Does not run real multi-GPU execution.
- Does not implement tensor parallelism.
- Does not implement pipeline parallelism.
- Does not measure NCCL communication.
- Does not measure latency or throughput.
- Does not model activation partitioning.
- Does not model optimizer-state partitioning.
- Does not model device imbalance.

## 11. Next Step

The next step is to combine model parallelism with total-memory approximation and compare it with quantization and sparsity.

## 12. Conclusion

ModelParallelMemoryEstimator makes the model-parallel part of the project reusable, modular, and easier to defend.