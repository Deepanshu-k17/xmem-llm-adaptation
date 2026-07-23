# QuantizationMemoryEstimator Report

## 1. Goal

This report documents the reusable QuantizationMemoryEstimator module added to the project.

The module converts the Day 42 and Day 43 quantization-memory formulas into reusable code under `src/estimators/`.

## 2. Module Path

`src/estimators/quantization_memory_estimator.py`

## 3. Supported Dtypes

- fp32
- fp16
- int8
- int4

## 4. Supported Metadata Cases

- none
- scale_only_fp16
- scale_plus_zero_point

## 5. What the Estimator Computes

The estimator computes:

- raw parameter memory
- grouped metadata memory
- effective parameter memory
- reduction compared with fp32
- metadata overhead as a percentage of raw quantized memory

## 6. Important Scope Clarification

This is an analytical estimator.

It does not run real int8/int4 quantized kernels.

It estimates parameter-memory behavior based on parameter count, dtype storage size, group size, and metadata assumptions.

## 7. Default Summary at Group Size 128

| model_name   | dtype   |   group_size |   raw_parameter_memory_MB |   metadata_memory_MB |   effective_parameter_memory_MB |   effective_reduction_percent |   metadata_overhead_percent_of_raw_quantized | scope_note                                                                  |
|:-------------|:--------|-------------:|--------------------------:|---------------------:|--------------------------------:|------------------------------:|---------------------------------------------:|:----------------------------------------------------------------------------|
| distilgpt2   | fp32    |          128 |                  312.472  |              0       |                        312.472  |                        0      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   | fp16    |          128 |                  156.236  |              0       |                        156.236  |                       50      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   | int8    |          128 |                   78.1179 |              2.44118 |                         80.5591 |                       74.2188 |                                        3.125 | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   | int4    |          128 |                   39.059  |              2.44118 |                         41.5001 |                       86.7188 |                                        6.25  | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         | fp32    |          128 |                  474.7    |              0       |                        474.7    |                        0      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         | fp16    |          128 |                  237.35   |              0       |                        237.35   |                       50      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         | int8    |          128 |                  118.675  |              3.7086  |                        122.384  |                       74.2188 |                                        3.125 | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         | int4    |          128 |                   59.3375 |              3.7086  |                         63.0461 |                       86.7188 |                                        6.25  | Analytical parameter-memory estimate, not real quantized runtime execution. |

## 8. Key Findings

|   finding_id | finding                                                                                        | evidence                                                                                                 | why_it_matters                                                              |
|-------------:|:-----------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
|            1 | QuantizationMemoryEstimator reproduces the Day 43 metadata-overhead simulation.                | At group size 128, int8 effective memory reduction is about 74.22% and int4 is about 86.72%.             | The quantization logic is now reusable instead of notebook-only.            |
|            2 | The estimator separates raw parameter memory, metadata memory, and effective parameter memory. | Each output row includes raw_parameter_memory_MB, metadata_memory_MB, and effective_parameter_memory_MB. | This avoids overclaiming quantization savings.                              |
|            3 | Metadata overhead remains proportionally larger for int4 than int8.                            | At group size 128, metadata overhead averages 3.125% for int8 and 6.250% for int4.                       | Lower-bit quantization is more sensitive to metadata assumptions.           |
|            4 | QuantizationMemoryEstimator is analytical, not runtime profiling.                              | The estimator uses parameter counts, bytes per parameter, group size, and metadata assumptions.          | The module should not be misrepresented as real int8/int4 CUDA measurement. |

## 9. Why This Module Matters

Before this module, quantization calculations existed only in notebook-level tables.

Now the project has a reusable quantization estimator that can be used by later reports, simulations, and estimator-comparison modules.

## 10. Limitations

- Does not measure real CUDA runtime memory.
- Does not include activation quantization.
- Does not include KV-cache quantization.
- Does not include kernel-specific packing/alignment overhead.
- Does not evaluate model accuracy or quality loss.
- Does not measure latency.

## 11. Next Step

The next step is to compare predicted memory under fp32, fp16, int8, and int4 using this module and connect it with the existing inference/training estimator results.

## 12. Conclusion

QuantizationMemoryEstimator makes the quantization-memory part of the project reusable, modular, and easier to defend.