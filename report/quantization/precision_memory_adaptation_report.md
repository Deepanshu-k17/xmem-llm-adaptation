# Precision Memory Adaptation Report

## 1. Goal

This report summarizes the fp32 vs fp16 precision-memory analysis for GPT-style inference workloads.

This is the first step in the precision and quantization adaptation phase of the xMem-inspired LLM memory profiling project.

## 2. Why Precision Matters

Precision directly affects tensor storage size.

fp32 uses 4 bytes per parameter value, while fp16 uses 2 bytes per parameter value.

Therefore, fp16 is expected to reduce parameter-related memory substantially.

However, total GPU memory is not reduced by exactly 50% because activations, framework overhead, CUDA allocator behavior, and non-parameter buffers also contribute.

## 3. Precision Comparison Results

| model_name   | task      |   batch_size |   input_tokens |   max_new_tokens |   fp32_peak_allocated_MB |   fp16_peak_allocated_MB |   fp32_peak_reserved_MB |   fp16_peak_reserved_MB | source_note                                         |   allocated_reduction_MB |   allocated_reduction_percent |   reserved_reduction_MB |   reserved_reduction_percent |
|:-------------|:----------|-------------:|---------------:|-----------------:|-------------------------:|-------------------------:|------------------------:|------------------------:|:----------------------------------------------------|-------------------------:|------------------------------:|------------------------:|-----------------------------:|
| distilgpt2   | inference |            1 |             64 |              128 |                   343.41 |                   181.25 |                     nan |                     nan | distilgpt2 fp32/fp16 inference precision experiment |                   162.16 |                       47.2205 |                     nan |                     nan      |
| gpt2         | inference |            1 |            128 |              128 |                   504.11 |                   264.78 |                     558 |                     268 | gpt2 fp32/fp16 inference validation                 |                   239.33 |                       47.4757 |                     290 |                      51.9713 |

## 4. Reduction Summary

| metric                              |   value | interpretation                                                                                         |
|:------------------------------------|--------:|:-------------------------------------------------------------------------------------------------------|
| average_allocated_reduction_percent | 47.3481 | Average peak allocated memory reduction from fp32 to fp16 across tested GPT-style inference workloads. |
| min_allocated_reduction_percent     | 47.2205 | Smallest observed allocated memory reduction from fp32 to fp16.                                        |
| max_allocated_reduction_percent     | 47.4757 | Largest observed allocated memory reduction from fp32 to fp16.                                         |
| gpt2_reserved_reduction_percent     | 51.9713 | Observed reserved memory reduction for gpt2 fp32 to fp16 inference.                                    |

## 5. Key Findings

|   finding_id | finding                                                                        | evidence                                                                                  | why_it_matters                                                                                       |
|-------------:|:-------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
|            1 | fp16 reduced peak allocated memory by around 47% for both distilgpt2 and gpt2. | distilgpt2 reduced from 343.41 MB to 181.25 MB; gpt2 reduced from 504.11 MB to 264.78 MB. | Precision is a major memory optimization lever for LLM inference workloads.                          |
|            2 | The fp16 reduction was consistent across model sizes.                          | distilgpt2 reduction was 47.22%, while gpt2 reduction was 47.48%.                         | This suggests that dtype-aware memory estimation can generalize across GPT-style models.             |
|            3 | fp16 does not always give exactly 50% total memory reduction.                  | Observed reductions were around 47%, not exactly 50%.                                     | Non-parameter memory, activations, allocator behavior, and framework overhead prevent exact halving. |
|            4 | Precision-aware estimation is required before quantization estimation.         | fp32 and fp16 memory behavior differs significantly.                                      | This prepares the project for int8/int4 quantization-memory simulation.                              |

## 6. Interpretation

For distilgpt2, fp16 reduced peak allocated memory from 343.41 MB to 181.25 MB, a reduction of about 47.22%.

For gpt2, fp16 reduced peak allocated memory from 504.11 MB to 264.78 MB, a reduction of about 47.48%.

The reduction is close to half but not exactly 50%, because total memory includes more than just model parameters.

This confirms that dtype-aware correction is necessary for accurate memory prediction.

## 7. Connection to Quantization

fp16 precision analysis provides the foundation for quantization-memory estimation.

After fp32 and fp16, the next step is to analytically estimate memory under int8 and int4 quantization.

Quantization is expected to reduce parameter memory further, but real runtime memory savings depend on kernel support, quantization implementation, and whether activations/optimizer states are also quantized.

## 8. Limitations

- Current precision comparison is focused on inference.
- Training fp16 behavior is not yet evaluated.
- Accuracy/quality impact of fp16 is not measured.
- Reserved memory can vary due to CUDA allocator behavior.
- int8/int4 quantization is not implemented yet; it will be added as an analytical estimator first.

## 9. Conclusion

fp16 provides a consistent memory reduction of about 47% for the tested GPT-style inference workloads.

This supports adding precision-aware and quantization-aware memory estimation modules to the project.