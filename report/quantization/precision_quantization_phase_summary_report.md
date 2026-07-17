# Precision and Quantization Phase Summary Report

## 1. Goal

This report summarizes the precision and quantization adaptation phase of the xMem-inspired LLM memory profiling project.

This phase extends the project beyond basic inference/training memory prediction toward optimization-aware memory estimation.

## 2. Phase Coverage

The phase covers:

- measured fp32 vs fp16 precision-memory behavior
- theoretical fp32/fp16/int8/int4 parameter-memory formulas
- quantization metadata overhead simulation
- reusable QuantizationMemoryEstimator module
- measured vs analytical precision comparison
- quantization-aware total-memory approximation

## 3. Phase Summary

| day    | phase_step                                    | main_output                                                                                    | main_result                                                                                           | scope                                                                |
|:-------|:----------------------------------------------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| Day 41 | fp32 vs fp16 precision analysis               | Measured fp16 runtime memory reduction for distilgpt2 and gpt2.                                | Average fp16 peak allocated memory reduction was 47.35%.                                              | Measured fp32/fp16 inference memory.                                 |
| Day 42 | Quantization memory theory                    | Theoretical parameter-memory formulas for fp32, fp16, int8, and int4.                          | Theoretical parameter reductions: fp16 50%, int8 75%, int4 87.5%.                                     | Analytical parameter-memory estimation only.                         |
| Day 43 | Metadata overhead simulation                  | Grouped quantization metadata overhead using scale + zero-point metadata.                      | At group size 128, int8 effective reduction was 74.22%, int4 was 86.72%.                              | Analytical parameter + metadata memory estimation.                   |
| Day 44 | QuantizationMemoryEstimator module            | Reusable estimator module under src/estimators/.                                               | Estimator computes raw parameter memory, metadata memory, effective memory, and reduction vs fp32.    | Reusable analytical estimator code.                                  |
| Day 45 | Measured vs analytical comparison             | Compared measured fp32/fp16 runtime memory with analytical parameter-memory estimates.         | Measured fp16 reduction averaged 47.35%, about 2.65 points below theoretical 50%.                     | Measured fp32/fp16 + analytical parameter-memory comparison.         |
| Day 46 | Quantization-aware total-memory approximation | Estimated total memory under fp16, int8, and int4 using parameter/non-parameter decomposition. | Estimated int8 total reduction: 67.53% distilgpt2, 69.89% gpt2; int4: 78.91% distilgpt2, 81.66% gpt2. | Analytical total-memory approximation, not real int8/int4 execution. |

## 4. Precision Results

| model_name   | task      |   batch_size |   input_tokens |   max_new_tokens |   fp32_peak_allocated_MB |   fp16_peak_allocated_MB |   fp32_peak_reserved_MB |   fp16_peak_reserved_MB | source_note                                         |   allocated_reduction_MB |   allocated_reduction_percent |   reserved_reduction_MB |   reserved_reduction_percent |
|:-------------|:----------|-------------:|---------------:|-----------------:|-------------------------:|-------------------------:|------------------------:|------------------------:|:----------------------------------------------------|-------------------------:|------------------------------:|------------------------:|-----------------------------:|
| distilgpt2   | inference |            1 |             64 |              128 |                   343.41 |                   181.25 |                     nan |                     nan | distilgpt2 fp32/fp16 inference precision experiment |                   162.16 |                       47.2205 |                     nan |                     nan      |
| gpt2         | inference |            1 |            128 |              128 |                   504.11 |                   264.78 |                     558 |                     268 | gpt2 fp32/fp16 inference validation                 |                   239.33 |                       47.4757 |                     290 |                      51.9713 |

## 5. Quantization Formats

| dtype   |   bits_per_parameter |   bytes_per_parameter |   relative_to_fp32 |   theoretical_parameter_reduction_percent | notes                                                                      |
|:--------|---------------------:|----------------------:|-------------------:|------------------------------------------:|:---------------------------------------------------------------------------|
| fp32    |                   32 |                   4   |              1     |                                       0   | Full precision baseline.                                                   |
| fp16    |                   16 |                   2   |              0.5   |                                      50   | Half precision; commonly used for inference/training acceleration.         |
| int8    |                    8 |                   1   |              0.25  |                                      75   | 8-bit quantization; usually needs scales/zero-points and kernel support.   |
| int4    |                    4 |                   0.5 |              0.125 |                                      87.5 | 4-bit quantization; highly compressed but implementation overhead matters. |

## 6. Effective Quantization Memory at Default Group Size

| model_name   | dtype   |   group_size |   raw_parameter_memory_MB |   metadata_memory_MB |   effective_parameter_memory_MB |   fp32_parameter_memory_MB |   effective_reduction_percent |   metadata_overhead_percent_of_raw_quantized |
|:-------------|:--------|-------------:|--------------------------:|---------------------:|--------------------------------:|---------------------------:|------------------------------:|---------------------------------------------:|
| distilgpt2   | fp32    |          128 |                  312.472  |              0       |                        312.472  |                    312.472 |                        0      |                                        0     |
| distilgpt2   | fp16    |          128 |                  156.236  |              0       |                        156.236  |                    312.472 |                       50      |                                        0     |
| distilgpt2   | int8    |          128 |                   78.1179 |              2.44118 |                         80.5591 |                    312.472 |                       74.2188 |                                        3.125 |
| distilgpt2   | int4    |          128 |                   39.059  |              2.44118 |                         41.5001 |                    312.472 |                       86.7188 |                                        6.25  |
| gpt2         | fp32    |          128 |                  474.7    |              0       |                        474.7    |                    474.7   |                        0      |                                        0     |
| gpt2         | fp16    |          128 |                  237.35   |              0       |                        237.35   |                    474.7   |                       50      |                                        0     |
| gpt2         | int8    |          128 |                  118.675  |              3.7086  |                        122.384  |                    474.7   |                       74.2187 |                                        3.125 |
| gpt2         | int4    |          128 |                   59.3375 |              3.7086  |                         63.0461 |                    474.7   |                       86.7187 |                                        6.25  |

## 7. Measured vs Analytical Comparison

| model_name   |   measured_fp32_peak_allocated_MB |   measured_fp16_peak_allocated_MB |   measured_allocated_reduction_percent |   analytical_fp32_parameter_memory_MB |   analytical_fp16_parameter_memory_MB |   analytical_parameter_reduction_percent |   gap_between_theoretical_and_measured_reduction_percent_points | interpretation                                                                                                                                                 |
|:-------------|----------------------------------:|----------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|-----------------------------------------:|----------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| distilgpt2   |                            343.41 |                            181.25 |                                47.2205 |                               312.472 |                               156.236 |                                       50 |                                                         2.77948 | Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters. |
| gpt2         |                            504.11 |                            264.78 |                                47.4757 |                               474.7   |                               237.35  |                                       50 |                                                         2.52425 | Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters. |

## 8. Total-memory Decomposition

| model_name   |   num_parameters |   measured_fp32_peak_allocated_MB |   fp32_parameter_memory_MB |   non_parameter_memory_MB |   parameter_fraction_percent |   non_parameter_fraction_percent | interpretation                                                                                                          |
|:-------------|-----------------:|----------------------------------:|---------------------------:|--------------------------:|-----------------------------:|---------------------------------:|:------------------------------------------------------------------------------------------------------------------------|
| distilgpt2   |         81912576 |                            343.41 |                    312.472 |                   30.9383 |                      90.9909 |                          9.00915 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |
| gpt2         |        124439808 |                            504.11 |                    474.7   |                   29.4098 |                      94.166  |                          5.83401 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |

## 9. Quantization-aware Total-memory Approximation

| model_name   | dtype   |   group_size |   effective_parameter_memory_MB |   non_parameter_memory_assumed_MB |   estimated_total_memory_MB |   estimated_total_reduction_percent |   measured_total_memory_MB |   measured_total_reduction_percent | estimation_type                | scope_note                                                                           |
|:-------------|:--------|-------------:|--------------------------------:|----------------------------------:|----------------------------:|------------------------------------:|---------------------------:|-----------------------------------:|:-------------------------------|:-------------------------------------------------------------------------------------|
| distilgpt2   | fp32    |          128 |                        312.472  |                           30.9383 |                    343.41   |                              0      |                     343.41 |                             0      | measured_available             | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   | fp16    |          128 |                        156.236  |                           30.9383 |                    187.174  |                             45.4954 |                     181.25 |                            47.2205 | measured_available             | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   | int8    |          128 |                         80.5591 |                           30.9383 |                    111.497  |                             67.5323 |                     nan    |                           nan      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| distilgpt2   | int4    |          128 |                         41.5001 |                           30.9383 |                     72.4385 |                             78.9061 |                     nan    |                           nan      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         | fp32    |          128 |                        474.7    |                           29.4098 |                    504.11   |                              0      |                     504.11 |                             0      | measured_available             | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         | fp16    |          128 |                        237.35   |                           29.4098 |                    266.76   |                             47.083  |                     264.78 |                            47.4757 | measured_available             | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         | int8    |          128 |                        122.384  |                           29.4098 |                    151.793  |                             69.8888 |                     nan    |                           nan      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |
| gpt2         | int4    |          128 |                         63.0461 |                           29.4098 |                     92.4559 |                             81.6596 |                     nan    |                           nan      | analytical_total_approximation | Total-memory approximation assumes non-parameter memory remains unchanged from fp32. |

## 10. Final Findings

|   finding_id | finding                                                                          | evidence                                                                                                           | interpretation                                                                                                           |
|-------------:|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
|            1 | fp16 significantly reduces measured inference memory, but not by exactly 50%.    | fp16 reduced peak allocated memory by 47.22% for distilgpt2 and 47.48% for gpt2.                                   | Total runtime memory contains parameters plus activations, KV cache, buffers, framework overhead, and allocator effects. |
|            2 | Parameter-memory reduction is not equal to total runtime-memory reduction.       | Theoretical fp16 parameter reduction is 50%, but measured runtime reduction averaged 47.35%.                       | Memory estimators must separate parameter and non-parameter components.                                                  |
|            3 | Quantization metadata reduces ideal int8/int4 savings.                           | At group size 128, int8 effective reduction became 74.22% instead of 75%, and int4 became 86.72% instead of 87.5%. | Scale and zero-point metadata must be included for realistic analytical quantization estimation.                         |
|            4 | int4 is more sensitive to metadata overhead than int8.                           | At group size 128, metadata overhead was 3.125% for int8 and 6.25% for int4.                                       | Lower-bit formats have smaller raw weight memory, so metadata becomes proportionally larger.                             |
|            5 | The QuantizationMemoryEstimator makes quantization logic reusable.               | The project now includes src/estimators/quantization_memory_estimator.py.                                          | Quantization estimation is now part of the project architecture, not only notebook analysis.                             |
|            6 | The total-memory approximation is more realistic than parameter-only estimation. | It decomposes measured fp32 runtime memory into parameter and non-parameter memory.                                | This gives a better estimate of practical memory savings under dtype/quantization changes.                               |
|            7 | int8 and int4 results are still analytical, not measured runtime profiling.      | No real int8/int4 quantized model was loaded or profiled.                                                          | The project should not overclaim quantized runtime results.                                                              |

## 11. Recommendations

| use_case                    | recommendation                                                  | reason                                                   | caution                                                                  |
|:----------------------------|:----------------------------------------------------------------|:---------------------------------------------------------|:-------------------------------------------------------------------------|
| Measured precision behavior | Use fp32/fp16 measured results from Day 41.                     | These are actual runtime CUDA memory measurements.       | Only covers distilgpt2 and gpt2 inference.                               |
| Parameter-memory estimation | Use QuantizationMemoryEstimator raw/effective parameter memory. | It includes dtype size and metadata overhead.            | Does not represent full runtime memory.                                  |
| Total-memory reasoning      | Use Day 46 quantization-aware total-memory approximation.       | It separates parameter and non-parameter memory.         | Assumes non-parameter memory remains unchanged.                          |
| int8/int4 claims            | Label as analytical estimates only.                             | No real int8/int4 quantized execution was measured.      | Do not claim runtime memory, latency, or quality results.                |
| Next technical phase        | Move to sparsity memory estimation.                             | Precision and quantization phase is now complete enough. | Do not keep over-tweaking quantization without new measured experiments. |

## 12. Main Interpretation

The precision and quantization phase shows that dtype and weight-compression techniques can significantly reduce memory.

However, parameter-memory reduction does not directly equal total runtime-memory reduction.

Measured fp16 runtime reduction averaged 47.35%, while theoretical fp16 parameter-memory reduction is 50%.

This difference appears because total memory includes non-parameter components such as activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

Adding metadata overhead makes int8/int4 estimates more realistic than raw bytes-per-parameter formulas.

The QuantizationMemoryEstimator module makes the quantization logic reusable inside the project.

## 13. Important Scope Clarification

fp32 and fp16 results include measured inference memory from earlier experiments.

int8 and int4 results are analytical estimates only.

No real int8/int4 quantized model execution was profiled.

## 14. Limitations

- Only distilgpt2 and gpt2 were used.
- int8/int4 values are analytical estimates only.
- No real quantized runtime execution was measured.
- Activation quantization is not modeled.
- KV-cache quantization is not modeled.
- Kernel packing/alignment overhead is not measured.
- Latency and quality impact are not evaluated.
- Non-parameter memory is assumed unchanged in total-memory approximation.

## 15. Conclusion

The precision and quantization adaptation phase is complete enough to support the official project direction.

The next phase should move to sparsity memory estimation.