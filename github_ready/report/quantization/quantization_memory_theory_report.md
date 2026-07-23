# Quantization Memory Theory Report

## 1. Goal

This report introduces the quantization-memory estimation phase of the xMem-inspired LLM memory profiling project.

The goal is to estimate how parameter memory changes under fp32, fp16, int8, and int4 storage formats.

## 2. Motivation

The official project scope includes optimization techniques such as quantization.

Quantization reduces the number of bits used to represent model weights, which can significantly reduce memory requirements for large models.

## 3. Quantization Formats

| dtype   |   bits_per_parameter |   bytes_per_parameter |   relative_to_fp32 |   theoretical_parameter_reduction_percent | notes                                                                      |
|:--------|---------------------:|----------------------:|-------------------:|------------------------------------------:|:---------------------------------------------------------------------------|
| fp32    |                   32 |                   4   |              1     |                                       0   | Full precision baseline.                                                   |
| fp16    |                   16 |                   2   |              0.5   |                                      50   | Half precision; commonly used for inference/training acceleration.         |
| int8    |                    8 |                   1   |              0.25  |                                      75   | 8-bit quantization; usually needs scales/zero-points and kernel support.   |
| int4    |                    4 |                   0.5 |              0.125 |                                      87.5 | 4-bit quantization; highly compressed but implementation overhead matters. |

## 4. Formula

Parameter memory is estimated as:

`parameter_memory_MB = num_parameters × bytes_per_parameter / 1024²`

This formula estimates only parameter storage, not full runtime memory.

## 5. Models Used

| model_name   |   num_parameters |   parameter_memory_fp32_MB_recorded | source                                                 |
|:-------------|-----------------:|------------------------------------:|:-------------------------------------------------------|
| distilgpt2   |         81912576 |                             312.472 | project model_config_utils / optimizer-state estimator |
| gpt2         |        124439808 |                             474.7   | project model_config_utils / gpt2 training validation  |

## 6. Parameter Memory Simulation

| model_name   |   num_parameters | dtype   |   bits_per_parameter |   bytes_per_parameter |   estimated_parameter_memory_MB |   fp32_parameter_memory_MB |   parameter_memory_reduction_MB |   parameter_memory_reduction_percent |   theoretical_reduction_percent |
|:-------------|-----------------:|:--------|---------------------:|----------------------:|--------------------------------:|---------------------------:|--------------------------------:|-------------------------------------:|--------------------------------:|
| distilgpt2   |         81912576 | fp32    |                   32 |                   4   |                        312.472  |                    312.472 |                       3.125e-07 |                          1.00009e-07 |                             0   |
| distilgpt2   |         81912576 | fp16    |                   16 |                   2   |                        156.236  |                    312.472 |                     156.236     |                         50           |                            50   |
| distilgpt2   |         81912576 | int8    |                    8 |                   1   |                         78.1179 |                    312.472 |                     234.354     |                         75           |                            75   |
| distilgpt2   |         81912576 | int4    |                    4 |                   0.5 |                         39.059  |                    312.472 |                     273.413     |                         87.5         |                            87.5 |
| gpt2         |        124439808 | fp32    |                   32 |                   4   |                        474.7    |                    474.7   |                      -3.125e-07 |                         -6.5831e-08  |                             0   |
| gpt2         |        124439808 | fp16    |                   16 |                   2   |                        237.35   |                    474.7   |                     237.35      |                         50           |                            50   |
| gpt2         |        124439808 | int8    |                    8 |                   1   |                        118.675  |                    474.7   |                     356.025     |                         75           |                            75   |
| gpt2         |        124439808 | int4    |                    4 |                   0.5 |                         59.3375 |                    474.7   |                     415.363     |                         87.5         |                            87.5 |

## 7. Reduction Summary

| dtype   |   avg_parameter_memory_MB |   avg_reduction_percent |   min_reduction_percent |   max_reduction_percent |
|:--------|--------------------------:|------------------------:|------------------------:|------------------------:|
| fp32    |                  393.586  |              1.7089e-08 |             -6.5831e-08 |             1.00009e-07 |
| fp16    |                  196.793  |             50          |             50          |            50           |
| int8    |                   98.3965 |             75          |             75          |            75           |
| int4    |                   49.1982 |             87.5        |             87.5        |            87.5         |

## 8. Key Findings

|   finding_id | finding                                                                                 | evidence                                                                                                    | why_it_matters                                                                                                            |
|-------------:|:----------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
|            1 | Parameter memory scales almost linearly with bytes per parameter.                       | fp16 uses 2 bytes/parameter, int8 uses 1 byte/parameter, and int4 uses 0.5 bytes/parameter.                 | This makes quantization a direct memory optimization lever for model weights.                                             |
|            2 | int8 theoretically reduces parameter memory by about 75% compared with fp32.            | int8 uses one-fourth the storage of fp32 for parameters.                                                    | This can significantly reduce inference memory for large models.                                                          |
|            3 | int4 theoretically reduces parameter memory by about 87.5% compared with fp32.          | int4 uses one-eighth the storage of fp32 for parameters.                                                    | This is useful for very memory-constrained deployment settings.                                                           |
|            4 | Total runtime memory will not reduce exactly by the theoretical parameter-memory ratio. | Day 41 showed fp16 allocated-memory reduction was about 47%, not exactly 50%.                               | Activations, KV cache, temporary buffers, metadata, scales, zero-points, and allocator behavior also affect total memory. |
|            5 | Quantization estimation should separate parameter memory from total GPU memory.         | Quantization mainly compresses weights, while activations and allocator overhead may remain fp16/fp32-like. | A realistic estimator must avoid overclaiming total memory savings.                                                       |

## 9. Interpretation

Parameter memory scales directly with bytes per parameter.

Compared with fp32, fp16 theoretically reduces parameter memory by 50%, int8 by 75%, and int4 by 87.5%.

However, total GPU memory does not reduce exactly by the same amount because activations, KV cache, temporary buffers, quantization metadata, framework overhead, and CUDA allocator behavior also contribute.

Day 41 already showed this effect: fp16 reduced peak allocated memory by about 47%, not exactly 50%.

## 10. Why This Is Analytical, Not Runtime Quantization Yet

This report does not claim that int8 or int4 models were actually loaded and executed.

It builds an analytical parameter-memory estimator that will later be connected to total memory estimation.

Actual runtime quantization depends on libraries, kernel support, metadata overhead, scales/zero-points, activation dtype, and hardware support.

## 11. Limitations

- This simulation estimates parameter memory only.
- It does not measure runtime latency or model quality.
- It does not include quantization metadata overhead yet.
- It does not include activation quantization.
- It does not include real int8/int4 execution.

## 12. Next Step

The next step is to build a QuantizationMemoryEstimator module that estimates parameter memory under different quantization formats and optionally includes metadata overhead.

## 13. Conclusion

Quantization provides a strong theoretical memory reduction for model weights.

This motivates adding quantization-aware memory estimation modules to extend xMem-style prediction for large model workloads.