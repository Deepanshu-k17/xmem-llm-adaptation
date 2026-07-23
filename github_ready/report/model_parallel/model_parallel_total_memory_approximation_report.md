# Model-parallel Total-memory Approximation Report

## 1. Goal

This report extends model-parallel parameter-memory estimation to total-memory approximation.

It also compares model parallelism with quantization and sparsity as memory-optimization strategies.

## 2. Method

Measured fp32 peak allocated memory is decomposed into parameter memory and non-parameter memory.

Then dense parameter memory is replaced with model-parallel per-device parameter memory.

The approximation is:

`estimated_total_per_device_memory = non_parameter_memory + model_parallel_per_device_parameter_memory`

## 3. Scope Clarification

This is an analytical approximation.

No real multi-GPU execution was performed.

No tensor parallelism, pipeline parallelism, NCCL communication, or distributed PyTorch runtime was profiled.

## 4. Memory Decomposition

| model_name   |   num_parameters |   measured_fp32_peak_allocated_MB |   dense_fp32_parameter_memory_MB |   non_parameter_memory_MB |   parameter_fraction_percent |   non_parameter_fraction_percent | assumption                                                               |
|:-------------|-----------------:|----------------------------------:|---------------------------------:|--------------------------:|-----------------------------:|---------------------------------:|:-------------------------------------------------------------------------|
| distilgpt2   |         81912576 |                            343.41 |                          312.472 |                   30.9383 |                      90.9909 |                          9.00915 | Non-parameter memory is assumed replicated and unchanged across devices. |
| gpt2         |        124439808 |                            504.11 |                          474.7   |                   29.4098 |                      94.166  |                          5.83401 | Non-parameter memory is assumed replicated and unchanged across devices. |

## 5. Model-parallel Total-memory Approximation

| model_name   | dtype   |   num_devices |   measured_fp32_total_memory_MB |   non_parameter_memory_assumed_MB |   total_parameter_memory_MB |   ideal_partitioned_parameter_memory_MB |   replicated_overhead_memory_MB |   communication_buffer_memory_MB |   model_parallel_per_device_parameter_memory_MB |   estimated_total_per_device_memory_MB |   ideal_parameter_reduction_percent |   effective_parameter_reduction_percent |   estimated_total_reduction_vs_fp32_percent | estimation_type                       | scope_note                                                                              |
|:-------------|:--------|--------------:|--------------------------------:|----------------------------------:|----------------------------:|----------------------------------------:|--------------------------------:|---------------------------------:|------------------------------------------------:|---------------------------------------:|------------------------------------:|----------------------------------------:|--------------------------------------------:|:--------------------------------------|:----------------------------------------------------------------------------------------|
| distilgpt2   | fp32    |             1 |                          343.41 |                           30.9383 |                     312.472 |                                312.472  |                        15.6236  |                          9.37415 |                                        337.469  |                               368.408  |                                 0   |                                    -8   |                                    -7.27927 | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp32    |             2 |                          343.41 |                           30.9383 |                     312.472 |                                156.236  |                        15.6236  |                          9.37415 |                                        181.234  |                               212.172  |                                50   |                                    42   |                                    38.2162  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp32    |             4 |                          343.41 |                           30.9383 |                     312.472 |                                 78.1179 |                        15.6236  |                          9.37415 |                                        103.116  |                               134.054  |                                75   |                                    67   |                                    60.9639  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp32    |             8 |                          343.41 |                           30.9383 |                     312.472 |                                 39.059  |                        15.6236  |                          9.37415 |                                         64.0567 |                                94.995  |                                87.5 |                                    79.5 |                                    72.3377  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp16    |             1 |                          343.41 |                           30.9383 |                     156.236 |                                156.236  |                         7.81179 |                          4.68708 |                                        168.735  |                               199.673  |                                 0   |                                    -8   |                                    41.8558  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp16    |             2 |                          343.41 |                           30.9383 |                     156.236 |                                 78.1179 |                         7.81179 |                          4.68708 |                                         90.6168 |                               121.555  |                                50   |                                    42   |                                    64.6035  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp16    |             4 |                          343.41 |                           30.9383 |                     156.236 |                                 39.059  |                         7.81179 |                          4.68708 |                                         51.5578 |                                82.4961 |                                75   |                                    67   |                                    75.9774  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| distilgpt2   | fp16    |             8 |                          343.41 |                           30.9383 |                     156.236 |                                 19.5295 |                         7.81179 |                          4.68708 |                                         32.0283 |                                62.9667 |                                87.5 |                                    79.5 |                                    81.6643  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp32    |             1 |                          504.11 |                           29.4098 |                     474.7   |                                474.7    |                        23.735   |                         14.241   |                                        512.676  |                               542.086  |                                 0   |                                    -8   |                                    -7.53328 | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp32    |             2 |                          504.11 |                           29.4098 |                     474.7   |                                237.35   |                        23.735   |                         14.241   |                                        275.326  |                               304.736  |                                50   |                                    42   |                                    39.5497  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp32    |             4 |                          504.11 |                           29.4098 |                     474.7   |                                118.675  |                        23.735   |                         14.241   |                                        156.651  |                               186.061  |                                75   |                                    67   |                                    63.0912  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp32    |             8 |                          504.11 |                           29.4098 |                     474.7   |                                 59.3375 |                        23.735   |                         14.241   |                                         97.3135 |                               126.723  |                                87.5 |                                    79.5 |                                    74.862   | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp16    |             1 |                          504.11 |                           29.4098 |                     237.35  |                                237.35   |                        11.8675  |                          7.1205  |                                        256.338  |                               285.748  |                                 0   |                                    -8   |                                    43.3164  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp16    |             2 |                          504.11 |                           29.4098 |                     237.35  |                                118.675  |                        11.8675  |                          7.1205  |                                        137.663  |                               167.073  |                                50   |                                    42   |                                    66.8579  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp16    |             4 |                          504.11 |                           29.4098 |                     237.35  |                                 59.3375 |                        11.8675  |                          7.1205  |                                         78.3255 |                               107.735  |                                75   |                                    67   |                                    78.6286  | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |
| gpt2         | fp16    |             8 |                          504.11 |                           29.4098 |                     237.35  |                                 29.6688 |                        11.8675  |                          7.1205  |                                         48.6568 |                                78.0666 |                                87.5 |                                    79.5 |                                    84.514   | analytical_total_memory_approximation | Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling. |

## 6. Comparison with Other Optimizations

| method                             | optimization_type             | runtime_measured   |   avg_total_reduction_percent |   distilgpt2_reduction_percent |   gpt2_reduction_percent | main_tradeoff                                                                  |
|:-----------------------------------|:------------------------------|:-------------------|------------------------------:|-------------------------------:|-------------------------:|:-------------------------------------------------------------------------------|
| fp16 precision                     | precision reduction           | partly measured    |                       46.2892 |                        45.4954 |                  47.083  | Practical and widely supported; measured fp16 behavior exists.                 |
| int8 quantization                  | weight quantization           | no                 |                       68.7105 |                        67.5323 |                  69.8888 | Strong analytical memory reduction; needs real quantized execution validation. |
| int4 quantization                  | weight quantization           | no                 |                       80.2829 |                        78.9061 |                  81.6596 | Largest analytical quantization saving; quality and kernel support matter.     |
| 75% unstructured sparsity          | sparse storage                | no                 |                       46.2892 |                        45.4954 |                  47.083  | Useful only after sparsity overcomes index metadata overhead.                  |
| 90% unstructured sparsity          | sparse storage                | no                 |                       74.0627 |                        72.7927 |                  75.3328 | Strong analytical saving; real sparse runtime depends on kernels/hardware.     |
| fp32 model parallelism - 2 devices | model parallelism             | no                 |                       38.8829 |                        38.2162 |                  39.5497 | Reduces per-device memory but needs multiple devices and communication.        |
| fp32 model parallelism - 4 devices | model parallelism             | no                 |                       62.0275 |                        60.9639 |                  63.0912 | Stronger per-device memory reduction; more communication and coordination.     |
| fp32 model parallelism - 8 devices | model parallelism             | no                 |                       73.5998 |                        72.3377 |                  74.862  | Large per-device reduction but higher distributed-system complexity.           |
| fp16 model parallelism - 4 devices | precision + model parallelism | no                 |                       77.303  |                        75.9774 |                  78.6286 | Combines dtype reduction and model partitioning; analytical only.              |

## 7. Key Findings

|   finding_id | finding                                                                                                    | evidence                                                                                                  | why_it_matters                                                                                 |
|-------------:|:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
|            1 | Model parallelism reduces per-device total memory by partitioning parameter memory.                        | Average fp32 total-memory reduction is 38.88% at 2 devices, 62.03% at 4 devices, and 73.60% at 8 devices. | This helps estimate whether a model can fit on smaller per-device memory budgets.              |
|            2 | Total-memory reduction is lower than parameter-only model-parallel reduction.                              | Non-parameter memory is assumed replicated and unchanged across devices.                                  | Activation, KV-cache, framework, and allocator memory limit total-memory scaling.              |
|            3 | Combining fp16 with model parallelism gives stronger estimated per-device reduction.                       | Average fp16 + 4-device model-parallel reduction is 77.30%.                                               | Multiple optimization techniques can be combined for larger memory savings.                    |
|            4 | Quantization can be more memory-efficient than low-device-count model parallelism in analytical estimates. | Average int8 reduction is 68.71%, while fp32 2-device model parallelism gives 38.88%.                     | Different optimization strategies solve different constraints.                                 |
|            5 | Model parallelism has a different tradeoff than quantization or sparsity.                                  | Model parallelism reduces per-device memory but requires multiple devices and communication.              | It is useful when a model is too large for one device, but adds distributed-system complexity. |
|            6 | This comparison is analytical, not runtime benchmarking.                                                   | Quantization int8/int4, sparsity, and model parallelism are not measured runtime executions.              | The project should present these as planning estimates, not deployment benchmarks.             |

## 8. Main Interpretation

Model parallelism reduces per-device memory by partitioning model parameters across devices.

However, total-memory reduction is lower than parameter-only reduction because non-parameter memory is assumed replicated.

Quantization and sparsity reduce memory differently.

Quantization reduces bytes per parameter.

Sparsity removes zero weights but requires index metadata.

Model parallelism reduces per-device parameter ownership but requires multiple devices and communication.

## 9. Limitations

- Analytical estimate only.
- No real multi-GPU execution.
- No tensor parallelism implementation.
- No pipeline parallelism implementation.
- No NCCL measurement.
- Non-parameter memory is assumed replicated and unchanged.
- Activation partitioning is not modeled.
- Optimizer-state partitioning is not modeled.
- Latency and throughput are not measured.

## 10. Next Step

The next step is to create a model-parallel phase summary and then move toward architecture comparison.

## 11. Conclusion

Day 54 shows that model parallelism is a useful per-device memory reduction technique, but it has a different tradeoff from quantization and sparsity.

It reduces per-device memory, but adds distributed-system complexity and communication overhead.