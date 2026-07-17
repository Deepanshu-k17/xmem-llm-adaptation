# Sparsity-aware Total Memory Approximation Report

## 1. Goal

This report extends the sparsity analysis from parameter-memory estimation to total-memory approximation.

The goal is to estimate how unstructured sparse parameter storage may affect total inference memory.

## 2. Method

The approximation decomposes measured fp32 peak allocated memory into:

- dense fp32 parameter memory
- non-parameter memory

Non-parameter memory includes activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

The total-memory approximation is:

`estimated_total_memory(sparsity) = measured_fp32_total_memory - dense_fp32_parameter_memory + sparse_total_parameter_memory(sparsity)`

Equivalently:

`estimated_total_memory(sparsity) = non_parameter_memory + sparse_total_parameter_memory(sparsity)`

## 3. Scope Clarification

This is an analytical approximation.

No real sparse model was loaded.

No sparse CUDA kernels were profiled.

## 4. Memory Decomposition

| model_name   |   num_parameters |   measured_fp32_peak_allocated_MB |   dense_fp32_parameter_memory_MB |   non_parameter_memory_MB |   parameter_fraction_percent |   non_parameter_fraction_percent | interpretation                                                                                                          |
|:-------------|-----------------:|----------------------------------:|---------------------------------:|--------------------------:|-----------------------------:|---------------------------------:|:------------------------------------------------------------------------------------------------------------------------|
| distilgpt2   |         81912576 |                            343.41 |                          312.472 |                   30.9383 |                      90.9909 |                          9.00915 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |
| gpt2         |        124439808 |                            504.11 |                          474.7   |                   29.4098 |                      94.166  |                          5.83401 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |

## 5. Sparsity-aware Total-memory Approximation

| model_name   |   sparsity_percent |   nonzero_fraction |   nonzero_parameters |   dense_fp32_parameter_memory_MB |   sparse_value_memory_MB |   sparse_index_memory_MB |   sparse_total_parameter_memory_MB |   non_parameter_memory_assumed_MB |   estimated_total_memory_MB |   measured_fp32_total_memory_MB |   estimated_total_reduction_percent | estimation_type                | scope_note                                                                           |
|:-------------|-------------------:|-------------------:|---------------------:|---------------------------------:|-------------------------:|-------------------------:|-----------------------------------:|----------------------------------:|----------------------------:|--------------------------------:|------------------------------------:|:-------------------------------|:-------------------------------------------------------------------------------------|
| distilgpt2   |                  0 |               1    |             81912576 |                          312.472 |                 312.472  |                 312.472  |                           624.943  |                           30.9383 |                    655.882  |                          343.41 |                            -90.9909 | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   |                 25 |               0.75 |             61434432 |                          312.472 |                 234.354  |                 234.354  |                           468.708  |                           30.9383 |                    499.646  |                          343.41 |                            -45.4954 | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   |                 50 |               0.5  |             40956288 |                          312.472 |                 156.236  |                 156.236  |                           312.472  |                           30.9383 |                    343.41   |                          343.41 |                              0      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   |                 75 |               0.25 |             20478144 |                          312.472 |                  78.1179 |                  78.1179 |                           156.236  |                           30.9383 |                    187.174  |                          343.41 |                             45.4954 | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   |                 90 |               0.1  |              8191257 |                          312.472 |                  31.2472 |                  31.2472 |                            62.4943 |                           30.9383 |                     93.4327 |                          343.41 |                             72.7927 | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         |                  0 |               1    |            124439808 |                          474.7   |                 474.7    |                 474.7    |                           949.4    |                           29.4098 |                    978.81   |                          504.11 |                            -94.166  | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         |                 25 |               0.75 |             93329856 |                          474.7   |                 356.025  |                 356.025  |                           712.05   |                           29.4098 |                    741.46   |                          504.11 |                            -47.083  | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         |                 50 |               0.5  |             62219904 |                          474.7   |                 237.35   |                 237.35   |                           474.7    |                           29.4098 |                    504.11   |                          504.11 |                              0      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         |                 75 |               0.25 |             31109952 |                          474.7   |                 118.675  |                 118.675  |                           237.35   |                           29.4098 |                    266.76   |                          504.11 |                             47.083  | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         |                 90 |               0.1  |             12443980 |                          474.7   |                  47.47   |                  47.47   |                            94.94   |                           29.4098 |                    124.35   |                          504.11 |                             75.3328 | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |

## 6. Key Findings

|   finding_id | finding                                                                                | evidence                                                                                                      | why_it_matters                                                                                             |
|-------------:|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
|            1 | Measured fp32 inference memory is parameter-dominated for the tested GPT-style models. | Average non-parameter memory fraction is about 7.42%.                                                         | Sparse parameter compression can significantly affect estimated total memory when sparsity is high enough. |
|            2 | Low sparsity can increase total memory under unstructured sparse storage.              | At 25% sparsity, average estimated total-memory reduction is -46.29%.                                         | Index metadata can outweigh savings when too many weights remain nonzero.                                  |
|            3 | 50% sparsity is approximately the total-memory break-even point in this model.         | At 50% sparsity, average estimated total-memory reduction is 0.00%.                                           | This matches the parameter-memory break-even behavior from Day 48/49.                                      |
|            4 | High sparsity gives meaningful total-memory savings.                                   | At 75% sparsity, average total-memory reduction is 46.29%; at 90%, it is 74.06%.                              | Sparsity only becomes useful when sparsity is high enough and sparse storage is supported.                 |
|            5 | This is still an analytical approximation, not real sparse runtime profiling.          | Sparse total memory is estimated using unchanged non-parameter memory and analytical sparse parameter memory. | Real sparse runtime depends on sparse kernels, hardware support, sparse format, and workload shape.        |

## 7. Interpretation

The tested GPT-style inference workloads are parameter-dominated.

This means sparse parameter compression can affect total memory when sparsity is high enough.

However, unstructured sparse storage needs index metadata.

At low sparsity, the index metadata can make sparse storage worse than dense storage.

At around 50% sparsity, this simple unstructured sparse format roughly breaks even.

At 75% and 90% sparsity, sparse storage begins to provide meaningful total-memory savings.

## 8. Limitations

- Analytical approximation only.
- No real sparse runtime execution.
- No sparse CUDA kernel profiling.
- Non-parameter memory is assumed unchanged.
- Structured sparsity is not modeled.
- Sparse matmul efficiency is not modeled.
- Latency and quality are not evaluated.
- Hardware-specific sparse acceleration is not measured.

## 9. Next Step

The next step is to summarize the sparsity phase and compare sparsity with quantization.

## 10. Conclusion

Day 50 shows that sparse memory benefits depend on both sparsity level and metadata overhead.

Unstructured sparsity becomes useful only after sparsity is high enough to overcome index metadata cost.