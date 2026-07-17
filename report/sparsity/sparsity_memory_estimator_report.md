# SparsityMemoryEstimator Report

## 1. Goal

This report documents the reusable SparsityMemoryEstimator module added to the project.

The module converts the Day 48 sparsity-memory formulas into reusable code under `src/estimators/`.

## 2. Module Path

`src/estimators/sparsity_memory_estimator.py`

## 3. Supported Storage Types

- dense_fp32
- unstructured_sparse_fp32

## 4. What the Estimator Computes

The estimator computes:

- dense fp32 parameter memory
- nonzero parameter count
- sparse value memory
- sparse index metadata memory
- sparse total parameter memory
- reduction compared with dense fp32

## 5. Important Scope Clarification

This is an analytical estimator.

It does not run real sparse CUDA kernels.

It estimates parameter-memory storage behavior based on parameter count, sparsity level, value bytes, and index bytes.

## 6. Estimator Demo

| model_name   |   num_parameters | storage_type             |   sparsity_percent |   nonzero_fraction |   nonzero_parameters |   value_bytes |   index_bytes_per_nonzero | uses_index_metadata   |   dense_fp32_parameter_memory_MB |   sparse_value_memory_MB |   sparse_index_memory_MB |   sparse_total_parameter_memory_MB |   sparse_reduction_percent |   sparse_overhead_vs_dense_percent | scope_note                                                                      |
|:-------------|-----------------:|:-------------------------|-------------------:|-------------------:|---------------------:|--------------:|--------------------------:|:----------------------|---------------------------------:|-------------------------:|-------------------------:|-----------------------------------:|---------------------------:|-----------------------------------:|:--------------------------------------------------------------------------------|
| distilgpt2   |         81912576 | unstructured_sparse_fp32 |                  0 |               1    |             81912576 |             4 |                         4 | True                  |                          312.472 |                 312.472  |                 312.472  |                           624.943  |                       -100 |                                100 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 | unstructured_sparse_fp32 |                 25 |               0.75 |             61434432 |             4 |                         4 | True                  |                          312.472 |                 234.354  |                 234.354  |                           468.708  |                        -50 |                                 50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 | unstructured_sparse_fp32 |                 50 |               0.5  |             40956288 |             4 |                         4 | True                  |                          312.472 |                 156.236  |                 156.236  |                           312.472  |                          0 |                                  0 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 | unstructured_sparse_fp32 |                 75 |               0.25 |             20478144 |             4 |                         4 | True                  |                          312.472 |                  78.1179 |                  78.1179 |                           156.236  |                         50 |                                -50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| distilgpt2   |         81912576 | unstructured_sparse_fp32 |                 90 |               0.1  |              8191257 |             4 |                         4 | True                  |                          312.472 |                  31.2472 |                  31.2472 |                            62.4943 |                         80 |                                -80 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 | unstructured_sparse_fp32 |                  0 |               1    |            124439808 |             4 |                         4 | True                  |                          474.7   |                 474.7    |                 474.7    |                           949.4    |                       -100 |                                100 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 | unstructured_sparse_fp32 |                 25 |               0.75 |             93329856 |             4 |                         4 | True                  |                          474.7   |                 356.025  |                 356.025  |                           712.05   |                        -50 |                                 50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 | unstructured_sparse_fp32 |                 50 |               0.5  |             62219904 |             4 |                         4 | True                  |                          474.7   |                 237.35   |                 237.35   |                           474.7    |                          0 |                                  0 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 | unstructured_sparse_fp32 |                 75 |               0.25 |             31109952 |             4 |                         4 | True                  |                          474.7   |                 118.675  |                 118.675  |                           237.35   |                         50 |                                -50 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |
| gpt2         |        124439808 | unstructured_sparse_fp32 |                 90 |               0.1  |             12443980 |             4 |                         4 | True                  |                          474.7   |                  47.47   |                  47.47   |                            94.94   |                         80 |                                -80 | Analytical sparse parameter-memory estimate, not real sparse runtime execution. |

## 7. Summary

|   sparsity_percent |   avg_sparse_total_parameter_memory_MB |   avg_sparse_reduction_percent |   min_sparse_reduction_percent |   max_sparse_reduction_percent |
|-------------------:|---------------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|
|                  0 |                               787.172  |                           -100 |                           -100 |                           -100 |
|                 25 |                               590.379  |                            -50 |                            -50 |                            -50 |
|                 50 |                               393.586  |                              0 |                              0 |                              0 |
|                 75 |                               196.793  |                             50 |                             50 |                             50 |
|                 90 |                                78.7172 |                             80 |                             80 |                             80 |

## 8. Key Findings

|   finding_id | finding                                                                    | evidence                                                                                          | why_it_matters                                                                 |
|-------------:|:---------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|
|            1 | SparsityMemoryEstimator reproduces the Day 48 analytical sparsity results. | 50% sparsity breaks even, 75% sparsity gives 50% reduction, and 90% sparsity gives 80% reduction. | Sparsity logic is now reusable project code instead of notebook-only analysis. |
|            2 | The estimator separates value memory and index metadata memory.            | Each output row includes sparse_value_memory_MB and sparse_index_memory_MB.                       | This prevents overclaiming sparse memory savings.                              |
|            3 | Unstructured sparsity has a high metadata cost.                            | At 0% sparsity, sparse storage uses 2x dense memory because it stores both values and indices.    | Sparse storage is not automatically better than dense storage.                 |
|            4 | SparsityMemoryEstimator is analytical, not runtime profiling.              | The estimator uses parameter count, sparsity level, value bytes, and index bytes.                 | The module should not be described as measured sparse CUDA behavior.           |

## 9. Why This Module Matters

Before this module, sparsity calculations existed only in notebook-level tables.

Now the project has a reusable sparsity estimator that can be used by later reports, simulations, and estimator-comparison modules.

## 10. Limitations

- Does not measure real sparse runtime memory.
- Does not run sparse matrix multiplication kernels.
- Does not measure latency.
- Does not model structured sparsity.
- Does not model hardware-specific sparse acceleration.
- Does not include activation memory or KV-cache memory.
- Does not evaluate accuracy or quality impact.

## 11. Next Step

The next step is to create a sparsity-aware total-memory approximation by combining dense runtime memory with analytical sparse parameter memory.

## 12. Conclusion

SparsityMemoryEstimator makes the sparsity-memory part of the project reusable, modular, and easier to defend.