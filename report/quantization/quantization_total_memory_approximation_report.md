# Quantization-aware Total Memory Approximation Report

## 1. Goal

This report extends the quantization analysis from parameter-memory estimation to total-memory approximation.

The goal is to estimate how fp16, int8, and int4 parameter compression may affect total inference memory.

## 2. Method

The approximation decomposes measured fp32 peak allocated memory into:

- fp32 parameter memory
- non-parameter memory

Non-parameter memory includes activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

The total-memory approximation is:

`estimated_total_memory(dtype) = measured_fp32_total_memory - fp32_parameter_memory + effective_parameter_memory(dtype)`

Equivalently:

`estimated_total_memory(dtype) = non_parameter_memory + effective_parameter_memory(dtype)`

## 3. Scope Clarification

fp32 and fp16 measured runtime values come from earlier inference experiments.

int8 and int4 total-memory values are analytical approximations.

They are not real measured quantized runtime results.

## 4. Memory Decomposition

| model_name   |   num_parameters |   measured_fp32_peak_allocated_MB |   fp32_parameter_memory_MB |   non_parameter_memory_MB |   parameter_fraction_percent |   non_parameter_fraction_percent | interpretation                                                                                                          |
|:-------------|-----------------:|----------------------------------:|---------------------------:|--------------------------:|-----------------------------:|---------------------------------:|:------------------------------------------------------------------------------------------------------------------------|
| distilgpt2   |         81912576 |                            343.41 |                    312.472 |                   30.9383 |                      90.9909 |                          9.00915 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |
| gpt2         |        124439808 |                            504.11 |                    474.7   |                   29.4098 |                      94.166  |                          5.83401 | Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior. |

## 5. Quantization-aware Total-memory Approximation

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

## 6. Key Findings

|   finding_id | finding                                                                            | evidence                                                                                                          | why_it_matters                                                                                     |
|-------------:|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
|            1 | Measured fp32 inference memory is parameter-dominated for these GPT-style models.  | Average non-parameter memory fraction is about 7.42%.                                                             | Quantizing parameters can significantly reduce total memory for these small GPT-style models.      |
|            2 | The total-memory approximation closely matches measured fp16 behavior.             | Estimated fp16 total-memory reduction averages 46.29%, while measured fp16 reduction averages 47.35%.             | This validates the decomposition approach as a reasonable approximation.                           |
|            3 | Estimated int8 total-memory reduction is lower than parameter-only int8 reduction. | Estimated int8 total-memory reduction averages 68.71%, while parameter-only effective reduction was about 74.22%. | Non-parameter memory limits total-memory savings.                                                  |
|            4 | Estimated int4 total-memory reduction is lower than parameter-only int4 reduction. | Estimated int4 total-memory reduction averages 80.28%, while parameter-only effective reduction was about 86.72%. | Even aggressive weight compression does not eliminate activation, KV-cache, and runtime overhead.  |
|            5 | This is still an analytical approximation, not real quantized runtime profiling.   | int8 and int4 totals are estimated using unchanged non-parameter memory from fp32.                                | The result is useful for reasoning, but should not be overclaimed as measured quantized execution. |

## 7. Interpretation

The measured fp32 memory for distilgpt2 and gpt2 is parameter-dominated.

This means parameter compression through fp16, int8, or int4 can significantly reduce estimated total memory.

However, total memory savings are lower than parameter-only savings because non-parameter memory remains.

The approximation closely matches measured fp16 behavior, which supports using the same decomposition logic to reason about int8 and int4.

## 8. Limitations

- int8 and int4 values are analytical approximations only.
- No real int8/int4 quantized model was loaded.
- Non-parameter memory is assumed unchanged from fp32.
- Activation quantization is not modeled.
- KV-cache quantization is not modeled.
- Runtime kernel overhead, packing, alignment, and allocator behavior are not measured.
- Latency and quality impact are not evaluated.

## 9. Next Step

The next step is to summarize the precision and quantization adaptation phase into a clean phase report.

## 10. Conclusion

Day 46 shows that total-memory estimation should separate parameter and non-parameter memory.

This gives a more realistic view of quantization benefits than parameter-memory reduction alone.