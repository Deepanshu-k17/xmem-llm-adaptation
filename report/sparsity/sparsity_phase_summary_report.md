# Sparsity Phase Summary Report

## 1. Goal

This report summarizes the sparsity memory estimation phase of the xMem-inspired LLM memory profiling project.

The phase extends the project toward optimization-aware memory estimation by analyzing how unstructured sparsity affects parameter memory and estimated total memory.

## 2. Phase Coverage

| day    | phase_step                                | main_output                                                                    | main_result                                                                                                      | scope                                                                     |
|:-------|:------------------------------------------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------|
| Day 48 | Sparsity memory theory                    | Analytical unstructured sparse parameter-memory formulas.                      | 50% sparsity breaks even; 75% gives 50% parameter-memory reduction; 90% gives 80% reduction.                     | Analytical parameter-memory estimation only.                              |
| Day 49 | SparsityMemoryEstimator module            | Reusable estimator under src/estimators/sparsity_memory_estimator.py.          | Module estimates dense memory, sparse value memory, index metadata, total sparse memory, and reduction vs dense. | Reusable analytical estimator code.                                       |
| Day 50 | Sparsity-aware total-memory approximation | Combined measured fp32 runtime memory with analytical sparse parameter memory. | 50% sparsity breaks even in total memory; 75% and 90% sparsity reduce total memory significantly.                | Analytical total-memory approximation, not real sparse runtime execution. |

## 3. Sparsity Formulas

| storage_type             |   value_bytes |   index_bytes_per_nonzero | uses_index_metadata   | note                                                                              |
|:-------------------------|--------------:|--------------------------:|:----------------------|:----------------------------------------------------------------------------------|
| dense_fp32               |             4 |                         0 | False                 | Dense fp32 baseline. Stores every parameter value.                                |
| unstructured_sparse_fp32 |             4 |                         4 | True                  | Stores only nonzero values plus index metadata. Analytical storage estimate only. |

## 4. Sparsity Parameter-memory Simulation

| model_name   |   num_parameters |   sparsity_percent |   nonzero_fraction |   nonzero_parameters |   dense_fp32_parameter_memory_MB |   sparse_value_memory_MB |   sparse_index_memory_MB |   sparse_total_parameter_memory_MB |   sparse_reduction_MB |   sparse_reduction_percent | scope_note                                                                      |
|:-------------|-----------------:|-------------------:|-------------------:|---------------------:|---------------------------------:|-------------------------:|-------------------------:|-----------------------------------:|----------------------:|---------------------------:|:--------------------------------------------------------------------------------|
| distilgpt2   |         81912576 |                  0 |               1    |             81912576 |                          312.472 |                 312.472  |                 312.472  |                           624.943  |              -312.472 |                       -100 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 |                 25 |               0.75 |             61434432 |                          312.472 |                 234.354  |                 234.354  |                           468.708  |              -156.236 |                        -50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 |                 50 |               0.5  |             40956288 |                          312.472 |                 156.236  |                 156.236  |                           312.472  |                 0     |                          0 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 |                 75 |               0.25 |             20478144 |                          312.472 |                  78.1179 |                  78.1179 |                           156.236  |               156.236 |                         50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 |                 90 |               0.1  |              8191257 |                          312.472 |                  31.2472 |                  31.2472 |                            62.4943 |               249.977 |                         80 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 |                  0 |               1    |            124439808 |                          474.7   |                 474.7    |                 474.7    |                           949.4    |              -474.7   |                       -100 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 |                 25 |               0.75 |             93329856 |                          474.7   |                 356.025  |                 356.025  |                           712.05   |              -237.35  |                        -50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 |                 50 |               0.5  |             62219904 |                          474.7   |                 237.35   |                 237.35   |                           474.7    |                 0     |                          0 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 |                 75 |               0.25 |             31109952 |                          474.7   |                 118.675  |                 118.675  |                           237.35   |               237.35  |                         50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 |                 90 |               0.1  |             12443980 |                          474.7   |                  47.47   |                  47.47   |                            94.94   |               379.76  |                         80 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |

## 5. SparsityMemoryEstimator Summary

|   sparsity_percent |   avg_sparse_total_parameter_memory_MB |   avg_sparse_reduction_percent |   min_sparse_reduction_percent |   max_sparse_reduction_percent |
|-------------------:|---------------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|
|                  0 |                               787.172  |                           -100 |                           -100 |                           -100 |
|                 25 |                               590.379  |                            -50 |                            -50 |                            -50 |
|                 50 |                               393.586  |                              0 |                              0 |                              0 |
|                 75 |                               196.793  |                             50 |                             50 |                             50 |
|                 90 |                                78.7172 |                             80 |                             80 |                             80 |

## 6. Sparsity Total-memory Approximation

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

## 7. Sparsity vs Quantization Comparison

| method                    | compression_type                      | runtime_measured   | analytical_or_measured                                             |   avg_total_memory_reduction_percent |   distilgpt2_reduction_percent |   gpt2_reduction_percent | main_tradeoff                                                                                     |
|:--------------------------|:--------------------------------------|:-------------------|:-------------------------------------------------------------------|-------------------------------------:|-------------------------------:|-------------------------:|:--------------------------------------------------------------------------------------------------|
| fp16 precision            | lower precision dtype                 | partly measured    | measured fp32/fp16 runtime memory + analytical total approximation |                              46.2892 |                        45.4954 |                  47.083  | Usually practical and widely supported, but savings are lower than parameter-only 50%.            |
| int8 quantization         | lower precision weight quantization   | no                 | analytical estimate only                                           |                              68.7105 |                        67.5323 |                  69.8888 | Strong memory reduction, but requires real quantized execution support to validate.               |
| int4 quantization         | lower precision weight quantization   | no                 | analytical estimate only                                           |                              80.2829 |                        78.9061 |                  81.6596 | Largest estimated memory reduction, but metadata, packing, quality, and kernel support matter.    |
| 75% unstructured sparsity | remove zero weights but store indices | no                 | analytical estimate only                                           |                              46.2892 |                        45.4954 |                  47.083  | Only useful after enough sparsity to overcome index metadata overhead.                            |
| 90% unstructured sparsity | remove zero weights but store indices | no                 | analytical estimate only                                           |                              74.0627 |                        72.7927 |                  75.3328 | Strong estimated reduction, but real sparse speed/memory depends heavily on hardware and kernels. |

## 8. Final Sparsity Findings

|   finding_id | finding                                                                             | evidence                                                                                                              | interpretation                                                                                   |
|-------------:|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
|            1 | Unstructured sparsity is not automatically memory-efficient.                        | At 0% and 25% sparsity, estimated total memory is worse than dense fp32 because each nonzero stores value plus index. | Sparsity needs high zero density or efficient sparse formats to become useful.                   |
|            2 | 50% sparsity is the break-even point in the current unstructured sparse fp32 model. | At 50% sparsity, estimated total-memory reduction is 0% for both distilgpt2 and gpt2.                                 | The index metadata cancels out the value-memory savings.                                         |
|            3 | 75% sparsity gives similar total-memory savings to fp16 in this approximation.      | Average 75% sparsity reduction is 46.29%, while average fp16 reduction is 46.29%.                                     | Sparse storage needs quite high sparsity to compete with simple dtype reduction.                 |
|            4 | 90% sparsity gives strong memory reduction but still needs runtime validation.      | Average 90% sparsity total-memory reduction is 74.06%.                                                                | The analytical estimate is promising, but real sparse execution depends on kernels and hardware. |
|            5 | Quantization is easier to reason about for memory than unstructured sparsity.       | Quantization reduces bytes per value, while unstructured sparsity also adds index metadata.                           | Sparsity has a more complicated memory tradeoff than dtype reduction.                            |
|            6 | The sparsity phase is analytical, not measured runtime profiling.                   | No sparse CUDA kernels, sparse matrix multiplications, or compressed model execution were measured.                   | The project should clearly label sparsity results as analytical estimates.                       |

## 9. Recommendations

| use_case                       | recommendation                                                 | reason                                                                             | caution                                                               |
|:-------------------------------|:---------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------|
| Safe project claim             | Describe sparsity results as analytical memory estimates only. | No real sparse runtime execution was measured.                                     | Do not claim sparse speedup or measured sparse CUDA memory.           |
| Memory optimization comparison | Present fp16 as the strongest measured precision result.       | fp16 has actual measured runtime memory data from earlier experiments.             | Only validated on distilgpt2 and gpt2 inference.                      |
| Sparsity interpretation        | Emphasize metadata overhead and break-even sparsity.           | The main systems insight is that sparsity can be worse than dense at low sparsity. | Avoid saying sparsity always reduces memory.                          |
| Future work                    | Mention real sparse kernel profiling as future work.           | Runtime behavior depends on sparse formats, hardware, and kernels.                 | Analytical estimates cannot prove real deployment behavior.           |
| Next project phase             | Move to model-parallel memory partitioning simulation.         | Precision, quantization, and sparsity adaptation phases are now complete enough.   | Do not overextend sparsity without real sparse execution experiments. |

## 10. Main Interpretation

The sparsity phase shows that unstructured sparsity is not automatically memory-efficient.

Unlike quantization, which reduces bytes per parameter value, unstructured sparsity also requires index metadata.

In the simple sparse fp32 format used here, 50% sparsity is approximately the break-even point.

Below 50% sparsity, sparse storage can use more memory than dense fp32.

At 75% and 90% sparsity, sparse storage begins to provide meaningful estimated total-memory savings.

However, these are analytical estimates only.

No real sparse runtime execution was measured.

## 11. Limitations

- Analytical estimates only.
- No real sparse CUDA kernel profiling.
- No sparse matrix multiplication benchmark.
- No latency measurement.
- No accuracy/quality measurement.
- Structured sparsity is not modeled.
- Hardware-specific sparse acceleration is not modeled.
- Non-parameter memory is assumed unchanged.

## 12. Conclusion

The sparsity phase is complete enough for the current project scope.

The main contribution is a reusable SparsityMemoryEstimator and a systems-level explanation of why sparse memory savings depend on sparsity level and metadata overhead.

The next phase should move to model-parallel memory partitioning simulation.