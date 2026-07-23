# Quantization Dtype Comparison Report

## 1. Goal

This report compares fp32, fp16, int8, and int4 memory estimates using the reusable QuantizationMemoryEstimator module.

It also compares measured fp32/fp16 inference memory against analytical parameter-memory estimates.

## 2. Important Scope Clarification

fp32 and fp16 values include measured inference memory from earlier experiments.

int8 and int4 values are analytical parameter-memory estimates with metadata overhead.

This report does not claim real int8/int4 runtime execution.

## 3. Quantization Dtype Comparison

| model_name   |   num_parameters | dtype   |   bits_per_parameter |   bytes_per_parameter |   group_size | metadata_case         |   raw_parameter_memory_MB |   metadata_memory_MB |   effective_parameter_memory_MB |   fp32_parameter_memory_MB |   effective_reduction_percent |   metadata_overhead_percent_of_raw_quantized | scope_note                                                                  |
|:-------------|-----------------:|:--------|---------------------:|----------------------:|-------------:|:----------------------|--------------------------:|---------------------:|--------------------------------:|---------------------------:|------------------------------:|---------------------------------------------:|:----------------------------------------------------------------------------|
| distilgpt2   |         81912576 | fp32    |                   32 |                   4   |          128 | scale_plus_zero_point |                  312.472  |              0       |                        312.472  |                    312.472 |                        0      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   |         81912576 | fp16    |                   16 |                   2   |          128 | scale_plus_zero_point |                  156.236  |              0       |                        156.236  |                    312.472 |                       50      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   |         81912576 | int8    |                    8 |                   1   |          128 | scale_plus_zero_point |                   78.1179 |              2.44118 |                         80.5591 |                    312.472 |                       74.2188 |                                        3.125 | Analytical parameter-memory estimate, not real quantized runtime execution. |
| distilgpt2   |         81912576 | int4    |                    4 |                   0.5 |          128 | scale_plus_zero_point |                   39.059  |              2.44118 |                         41.5001 |                    312.472 |                       86.7188 |                                        6.25  | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         |        124439808 | fp32    |                   32 |                   4   |          128 | scale_plus_zero_point |                  474.7    |              0       |                        474.7    |                    474.7   |                        0      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         |        124439808 | fp16    |                   16 |                   2   |          128 | scale_plus_zero_point |                  237.35   |              0       |                        237.35   |                    474.7   |                       50      |                                        0     | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         |        124439808 | int8    |                    8 |                   1   |          128 | scale_plus_zero_point |                  118.675  |              3.7086  |                        122.384  |                    474.7   |                       74.2188 |                                        3.125 | Analytical parameter-memory estimate, not real quantized runtime execution. |
| gpt2         |        124439808 | int4    |                    4 |                   0.5 |          128 | scale_plus_zero_point |                   59.3375 |              3.7086  |                         63.0461 |                    474.7   |                       86.7188 |                                        6.25  | Analytical parameter-memory estimate, not real quantized runtime execution. |

## 4. Measured vs Analytical Comparison

| model_name   |   measured_fp32_peak_allocated_MB |   measured_fp16_peak_allocated_MB |   measured_allocated_reduction_percent |   analytical_fp32_parameter_memory_MB |   analytical_fp16_parameter_memory_MB |   analytical_parameter_reduction_percent |   gap_between_theoretical_and_measured_reduction_percent_points | interpretation                                                                                                                                                 |
|:-------------|----------------------------------:|----------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|-----------------------------------------:|----------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| distilgpt2   |                            343.41 |                            181.25 |                                47.2205 |                               312.472 |                               156.236 |                                       50 |                                                         2.77948 | Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters. |
| gpt2         |                            504.11 |                            264.78 |                                47.4757 |                               474.7   |                               237.35  |                                       50 |                                                         2.52425 | Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters. |

## 5. Key Findings

|   finding_id | finding                                                                                               | evidence                                                                                         | why_it_matters                                                                                                  |
|-------------:|:------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
|            1 | Measured fp16 runtime memory reduction is slightly lower than theoretical parameter-memory reduction. | Measured allocated reduction averaged 47.35%, while theoretical fp16 parameter reduction is 50%. | Runtime memory includes activations, KV cache, buffers, framework overhead, and allocator behavior.             |
|            2 | The gap between theoretical and measured fp16 reduction is small but important.                       | Average gap was about 2.65 percentage points.                                                    | This shows why quantization estimators should not claim total memory savings equal to parameter-memory savings. |
|            3 | int8 and int4 provide strong analytical parameter-memory reductions after metadata overhead.          | At group size 128, int8 effective reduction averaged 74.22%, and int4 averaged 86.72%.           | Quantization is a strong candidate optimization for memory-constrained inference.                               |
|            4 | int8/int4 results remain analytical estimates.                                                        | The project estimates parameter and metadata memory but does not run real quantized kernels.     | This prevents overclaiming measured runtime quantization performance.                                           |
|            5 | Quantization affects parameter memory more directly than total runtime memory.                        | Measured fp16 reduction was below the 50% parameter-only expectation.                            | Future total-memory estimators should model activations and KV cache separately.                                |

## 6. Interpretation

The QuantizationMemoryEstimator predicts strong parameter-memory reductions for fp16, int8, and int4.

At group size 128, int8 gives about 74.22% effective parameter-memory reduction after metadata overhead, while int4 gives about 86.72%.

Measured fp16 runtime memory reduction is about 47%, slightly below the theoretical 50% parameter-memory reduction.

This gap exists because total runtime memory contains more than model parameters.

## 7. Why This Matters

Quantization mainly compresses weights.

However, real GPU memory also includes activations, KV cache, temporary buffers, metadata, framework overhead, and allocator behavior.

Therefore, a realistic memory estimator must distinguish parameter-memory savings from total runtime-memory savings.

## 8. Limitations

- int8/int4 are analytical estimates only.
- No real quantized model was loaded or profiled.
- No latency or quality evaluation was performed.
- Activation and KV-cache quantization are not modeled yet.
- CUDA allocator and kernel-specific packing overhead are not measured.

## 9. Next Step

The next step is to create a quantization-aware total-memory approximation that combines measured runtime memory structure with analytical parameter-memory reductions.

## 10. Conclusion

Day 45 connects the analytical quantization estimator with measured fp32/fp16 inference results.

This makes the quantization section more realistic because it clearly separates measured runtime memory from analytical parameter-memory estimation.