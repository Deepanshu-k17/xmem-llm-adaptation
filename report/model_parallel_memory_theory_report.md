# Model-parallel Memory Theory Report

## 1. Goal

This report starts the model-parallel memory estimation phase of the xMem-inspired LLM memory profiling project.

The goal is to analytically estimate how splitting model parameters across multiple devices affects per-device memory.

## 2. Important Scope Clarification

This is analytical model-parallel memory estimation only.

No real multi-GPU execution was performed.

No tensor parallelism, pipeline parallelism, NCCL communication, or distributed runtime was profiled.

## 3. Formulas

| concept                            | formula                                                                                 | meaning                                                                                |
|:-----------------------------------|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| single_device_parameter_memory     | total_parameter_memory_MB                                                               | Without model parallelism, each device stores the full parameter memory.               |
| ideal_partitioned_parameter_memory | total_parameter_memory_MB / num_devices                                                 | Ideal per-device parameter memory if parameters are evenly split.                      |
| replicated_overhead_memory         | total_parameter_memory_MB * replication_overhead_percent                                | Extra replicated memory such as small shared tensors, metadata, or framework overhead. |
| communication_buffer_memory        | total_parameter_memory_MB * communication_buffer_percent                                | Temporary memory used for communication buffers during distributed execution.          |
| estimated_per_device_memory        | partitioned_parameter_memory + replicated_overhead_memory + communication_buffer_memory | Simple analytical estimate of per-device model-parallel memory.                        |

## 4. Assumptions

- Replication overhead: 5.0%
- Communication buffer overhead: 3.0%
- Combined overhead: 8.0%
- Parameters are assumed to be evenly partitioned across devices.
- Overhead is estimated as a percentage of total parameter memory.

## 5. Models Used

| model_name   |   num_parameters |   fp32_parameter_memory_MB |   fp16_parameter_memory_MB |
|:-------------|-----------------:|---------------------------:|---------------------------:|
| distilgpt2   |         81912576 |                    312.472 |                    156.236 |
| gpt2         |        124439808 |                    474.7   |                    237.35  |

## 6. Model-parallel Parameter Partitioning Simulation

| model_name   | dtype   |   num_devices |   total_parameter_memory_MB |   ideal_partitioned_parameter_memory_MB |   replicated_overhead_memory_MB |   communication_buffer_memory_MB |   total_overhead_memory_MB |   estimated_per_device_memory_MB |   ideal_reduction_percent |   effective_reduction_percent |   overhead_fraction_of_total_percent | scope_note                                                                 |
|:-------------|:--------|--------------:|----------------------------:|----------------------------------------:|--------------------------------:|---------------------------------:|---------------------------:|---------------------------------:|--------------------------:|------------------------------:|-------------------------------------:|:---------------------------------------------------------------------------|
| distilgpt2   | fp32    |             1 |                     312.472 |                                312.472  |                        15.6236  |                          9.37415 |                    24.9977 |                         337.469  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp32    |             2 |                     312.472 |                                156.236  |                        15.6236  |                          9.37415 |                    24.9977 |                         181.234  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp32    |             4 |                     312.472 |                                 78.1179 |                        15.6236  |                          9.37415 |                    24.9977 |                         103.116  |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp32    |             8 |                     312.472 |                                 39.059  |                        15.6236  |                          9.37415 |                    24.9977 |                          64.0567 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp16    |             1 |                     156.236 |                                156.236  |                         7.81179 |                          4.68708 |                    12.4989 |                         168.735  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp16    |             2 |                     156.236 |                                 78.1179 |                         7.81179 |                          4.68708 |                    12.4989 |                          90.6168 |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp16    |             4 |                     156.236 |                                 39.059  |                         7.81179 |                          4.68708 |                    12.4989 |                          51.5578 |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| distilgpt2   | fp16    |             8 |                     156.236 |                                 19.5295 |                         7.81179 |                          4.68708 |                    12.4989 |                          32.0283 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp32    |             1 |                     474.7   |                                474.7    |                        23.735   |                         14.241   |                    37.976  |                         512.676  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp32    |             2 |                     474.7   |                                237.35   |                        23.735   |                         14.241   |                    37.976  |                         275.326  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp32    |             4 |                     474.7   |                                118.675  |                        23.735   |                         14.241   |                    37.976  |                         156.651  |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp32    |             8 |                     474.7   |                                 59.3375 |                        23.735   |                         14.241   |                    37.976  |                          97.3135 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp16    |             1 |                     237.35  |                                237.35   |                        11.8675  |                          7.1205  |                    18.988  |                         256.338  |                       0   |                          -8   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp16    |             2 |                     237.35  |                                118.675  |                        11.8675  |                          7.1205  |                    18.988  |                         137.663  |                      50   |                          42   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp16    |             4 |                     237.35  |                                 59.3375 |                        11.8675  |                          7.1205  |                    18.988  |                          78.3255 |                      75   |                          67   |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |
| gpt2         | fp16    |             8 |                     237.35  |                                 29.6688 |                        11.8675  |                          7.1205  |                    18.988  |                          48.6568 |                      87.5 |                          79.5 |                                    8 | Analytical model-parallel memory estimate, not real distributed execution. |

## 7. Key Findings

|   finding_id | finding                                                                                             | evidence                                                                                            | why_it_matters                                                                                    |
|-------------:|:----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
|            1 | Model parallelism reduces per-device parameter memory by partitioning model weights across devices. | Ideal partitioned memory scales as total_parameter_memory / num_devices.                            | This allows larger models to fit when one device cannot hold the full model.                      |
|            2 | Effective memory reduction is lower than ideal scaling because of overhead.                         | The simulation includes 8.0% combined replication and communication overhead.                       | Real model parallelism is not perfectly linear.                                                   |
|            3 | Two-device partitioning gives useful but limited memory reduction.                                  | Average fp32 effective per-device reduction at 2 devices is 42.00%.                                 | Two devices help, but overhead still consumes a visible part of memory.                           |
|            4 | Four and eight devices provide stronger per-device memory reduction.                                | Average fp32 effective reduction is 67.00% at 4 devices and 79.50% at 8 devices.                    | More devices reduce per-device weight storage, but communication and replication costs remain.    |
|            5 | This is analytical model-parallel planning, not real distributed profiling.                         | No tensor parallelism, pipeline parallelism, NCCL communication, or multi-GPU runtime was executed. | The results should be presented as memory-estimation logic, not measured distributed performance. |

## 8. Main Interpretation

Model parallelism reduces per-device parameter memory by partitioning model weights across devices.

In the ideal case, per-device parameter memory scales as total parameter memory divided by the number of devices.

However, practical systems have replication overhead and communication buffers.

This means effective per-device memory reduction is lower than ideal scaling.

## 9. Limitations

- Analytical estimate only.
- No real multi-GPU execution.
- No tensor parallelism implementation.
- No pipeline parallelism implementation.
- No NCCL communication measurement.
- No latency or throughput measurement.
- No activation partitioning modeled.
- No optimizer-state partitioning modeled.
- No device imbalance modeled.

## 10. Next Step

The next step is to create a reusable ModelParallelMemoryEstimator module under `src/estimators/`.

## 11. Conclusion

Day 52 shows the basic memory tradeoff of model parallelism.

Per-device memory decreases as parameters are partitioned across more devices, but overhead prevents perfect scaling.