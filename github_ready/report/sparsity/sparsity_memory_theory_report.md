# Sparsity Memory Theory Report

## 1. Goal

This report starts the sparsity memory estimation phase of the xMem-inspired LLM memory profiling project.

The goal is to analytically estimate how unstructured sparsity may affect parameter memory.

## 2. Important Scope Clarification

This is analytical parameter-memory estimation only.

No real sparse model was executed.

No sparse CUDA kernels were profiled.

The results should not be described as measured sparse runtime memory.

## 3. Sparsity Formats

| storage_type             |   value_bytes |   index_bytes_per_nonzero | uses_index_metadata   | note                                                                              |
|:-------------------------|--------------:|--------------------------:|:----------------------|:----------------------------------------------------------------------------------|
| dense_fp32               |             4 |                         0 | False                 | Dense fp32 baseline. Stores every parameter value.                                |
| unstructured_sparse_fp32 |             4 |                         4 | True                  | Stores only nonzero values plus index metadata. Analytical storage estimate only. |

## 4. Method

Dense fp32 parameter memory is estimated as:

`dense_parameter_memory_MB = num_parameters × 4 / 1024²`

For unstructured sparse storage, the estimate stores only nonzero values plus index metadata:

`nonzero_parameters = num_parameters × (1 - sparsity)`

`sparse_total_memory_MB = nonzero_parameters × (value_bytes + index_bytes) / 1024²`

## 5. Models Used

| model_name   |   num_parameters |   dense_fp32_parameter_memory_MB | source                                                 |
|:-------------|-----------------:|---------------------------------:|:-------------------------------------------------------|
| distilgpt2   |         81912576 |                          312.472 | project model_config_utils / optimizer-state estimator |
| gpt2         |        124439808 |                          474.7   | project model_config_utils / gpt2 training validation  |

## 6. Sparse Parameter-memory Simulation

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

## 7. Reduction Summary

|   sparsity_percent |   avg_sparse_total_parameter_memory_MB |   avg_reduction_percent |   min_reduction_percent |   max_reduction_percent |
|-------------------:|---------------------------------------:|------------------------:|------------------------:|------------------------:|
|                  0 |                               787.172  |                    -100 |                    -100 |                    -100 |
|                 25 |                               590.379  |                     -50 |                     -50 |                     -50 |
|                 50 |                               393.586  |                       0 |                       0 |                       0 |
|                 75 |                               196.793  |                      50 |                      50 |                      50 |
|                 90 |                                78.7172 |                      80 |                      80 |                      80 |

## 8. Key Findings

|   finding_id | finding                                                                        | evidence                                                                                                | why_it_matters                                                                                               |
|-------------:|:-------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
|            1 | Unstructured sparsity needs index metadata.                                    | The sparse estimate stores both nonzero values and index metadata.                                      | Sparse memory savings are lower than the raw zero-weight percentage suggests.                                |
|            2 | Low sparsity can use more memory than dense storage.                           | With value_bytes=4 and index_bytes=4, sparse storage at 25% sparsity is larger than dense fp32 storage. | Sparsity is not automatically a memory win unless sparsity is high enough or the sparse format is efficient. |
|            3 | Around 50% sparsity is the break-even point in this simple unstructured model. | At 50% sparsity, nonzero values plus indices roughly equal dense fp32 memory.                           | Below this level, sparse storage overhead can cancel the benefit of zero weights.                            |
|            4 | High sparsity can reduce parameter memory significantly.                       | At 75% and 90% sparsity, sparse parameter memory becomes lower than dense parameter memory.             | Sparsity can be useful for memory reduction when sparsity is high and sparse storage is supported.           |
|            5 | This is analytical storage estimation, not real sparse runtime profiling.      | The simulation does not run sparse CUDA kernels or measure latency.                                     | Sparse runtime benefits depend on hardware, kernels, sparse format, and workload shape.                      |

## 9. Interpretation

Sparsity does not automatically reduce memory.

Unstructured sparse storage needs index metadata, and this metadata can cancel out savings at low sparsity levels.

In this simple model, 50% sparsity is roughly the break-even point because each nonzero stores both a value and an index.

At higher sparsity levels such as 75% and 90%, sparse storage begins to provide significant parameter-memory savings.

## 10. Limitations

- This is an analytical estimate only.
- It does not run real sparse kernels.
- It does not measure latency.
- It does not model structured sparsity.
- It does not model sparse matrix multiplication efficiency.
- It does not include activation memory or KV-cache memory.
- It does not include hardware-specific sparse acceleration.

## 11. Next Step

The next step is to create a reusable SparsityMemoryEstimator module under `src/estimators/`.

## 12. Conclusion

Day 48 shows that sparsity memory savings depend heavily on metadata overhead and sparsity level.

Unlike quantization, sparsity is not automatically beneficial at low sparsity levels.