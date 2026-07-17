# Quantization Metadata Overhead Report

## 1. Goal

This report extends the Day 42 quantization-memory theory by adding metadata overhead assumptions.

Day 42 estimated raw parameter memory using bytes per parameter.

Day 43 adds scale and zero-point metadata overhead for grouped int8 and int4 quantization.

## 2. Important Scope Clarification

This is still an analytical simulation.

It does not claim real int8/int4 quantized model execution.

The goal is to make quantization-memory estimation more realistic by accounting for metadata overhead.

## 3. Metadata Assumptions

| metadata_case         |   scale_bytes |   zero_point_bytes |   metadata_bytes_per_group | note                                               |
|:----------------------|--------------:|-------------------:|---------------------------:|:---------------------------------------------------|
| scale_only_fp16       |             2 |                  0 |                          2 | Stores only fp16 scale per group.                  |
| scale_plus_zero_point |             2 |                  2 |                          4 | Stores fp16 scale and 2-byte zero-point per group. |

Main simulation case:

- metadata case: scale_plus_zero_point
- metadata bytes per group: 4.0
- group sizes tested: [32, 64, 128, 256]

## 4. Method

For fp32 and fp16, no quantization metadata is added.

For int8 and int4, metadata memory is estimated as:

`metadata_memory_MB = ceil(num_parameters / group_size) × metadata_bytes_per_group / 1024²`

Effective parameter memory is estimated as:

`effective_parameter_memory_MB = raw_parameter_memory_MB + metadata_memory_MB`

## 5. Full Metadata Simulation

| model_name   |   num_parameters | dtype   |   group_size | metadata_case         |   bytes_per_parameter |   raw_parameter_memory_MB |   metadata_memory_MB |   effective_parameter_memory_MB |   fp32_parameter_memory_MB |   effective_reduction_MB |   effective_reduction_percent |   metadata_overhead_percent_of_raw_quantized |   num_groups |
|:-------------|-----------------:|:--------|-------------:|:----------------------|----------------------:|--------------------------:|---------------------:|--------------------------------:|---------------------------:|-------------------------:|------------------------------:|---------------------------------------------:|-------------:|
| distilgpt2   |         81912576 | fp32    |           32 | scale_plus_zero_point |                   4   |                  312.472  |              0       |                        312.472  |                    312.472 |                3.125e-07 |                   1.00009e-07 |                                       0      |            0 |
| distilgpt2   |         81912576 | fp32    |           64 | scale_plus_zero_point |                   4   |                  312.472  |              0       |                        312.472  |                    312.472 |                3.125e-07 |                   1.00009e-07 |                                       0      |            0 |
| distilgpt2   |         81912576 | fp32    |          128 | scale_plus_zero_point |                   4   |                  312.472  |              0       |                        312.472  |                    312.472 |                3.125e-07 |                   1.00009e-07 |                                       0      |            0 |
| distilgpt2   |         81912576 | fp32    |          256 | scale_plus_zero_point |                   4   |                  312.472  |              0       |                        312.472  |                    312.472 |                3.125e-07 |                   1.00009e-07 |                                       0      |            0 |
| distilgpt2   |         81912576 | fp16    |           32 | scale_plus_zero_point |                   2   |                  156.236  |              0       |                        156.236  |                    312.472 |              156.236     |                  50           |                                       0      |            0 |
| distilgpt2   |         81912576 | fp16    |           64 | scale_plus_zero_point |                   2   |                  156.236  |              0       |                        156.236  |                    312.472 |              156.236     |                  50           |                                       0      |            0 |
| distilgpt2   |         81912576 | fp16    |          128 | scale_plus_zero_point |                   2   |                  156.236  |              0       |                        156.236  |                    312.472 |              156.236     |                  50           |                                       0      |            0 |
| distilgpt2   |         81912576 | fp16    |          256 | scale_plus_zero_point |                   2   |                  156.236  |              0       |                        156.236  |                    312.472 |              156.236     |                  50           |                                       0      |            0 |
| distilgpt2   |         81912576 | int8    |           32 | scale_plus_zero_point |                   1   |                   78.1179 |              9.76474 |                         87.8827 |                    312.472 |              224.589     |                  71.875       |                                      12.5    |      2559768 |
| distilgpt2   |         81912576 | int8    |           64 | scale_plus_zero_point |                   1   |                   78.1179 |              4.88237 |                         83.0003 |                    312.472 |              229.471     |                  73.4375      |                                       6.25   |      1279884 |
| distilgpt2   |         81912576 | int8    |          128 | scale_plus_zero_point |                   1   |                   78.1179 |              2.44118 |                         80.5591 |                    312.472 |              231.913     |                  74.2188      |                                       3.125  |       639942 |
| distilgpt2   |         81912576 | int8    |          256 | scale_plus_zero_point |                   1   |                   78.1179 |              1.22059 |                         79.3385 |                    312.472 |              233.133     |                  74.6094      |                                       1.5625 |       319971 |
| distilgpt2   |         81912576 | int4    |           32 | scale_plus_zero_point |                   0.5 |                   39.059  |              9.76474 |                         48.8237 |                    312.472 |              263.648     |                  84.375       |                                      25      |      2559768 |
| distilgpt2   |         81912576 | int4    |           64 | scale_plus_zero_point |                   0.5 |                   39.059  |              4.88237 |                         43.9413 |                    312.472 |              268.53      |                  85.9375      |                                      12.5    |      1279884 |
| distilgpt2   |         81912576 | int4    |          128 | scale_plus_zero_point |                   0.5 |                   39.059  |              2.44118 |                         41.5001 |                    312.472 |              270.972     |                  86.7188      |                                       6.25   |       639942 |
| distilgpt2   |         81912576 | int4    |          256 | scale_plus_zero_point |                   0.5 |                   39.059  |              1.22059 |                         40.2796 |                    312.472 |              272.192     |                  87.1094      |                                       3.125  |       319971 |
| gpt2         |        124439808 | fp32    |           32 | scale_plus_zero_point |                   4   |                  474.7    |              0       |                        474.7    |                    474.7   |               -3.125e-07 |                  -6.5831e-08  |                                       0      |            0 |
| gpt2         |        124439808 | fp32    |           64 | scale_plus_zero_point |                   4   |                  474.7    |              0       |                        474.7    |                    474.7   |               -3.125e-07 |                  -6.5831e-08  |                                       0      |            0 |
| gpt2         |        124439808 | fp32    |          128 | scale_plus_zero_point |                   4   |                  474.7    |              0       |                        474.7    |                    474.7   |               -3.125e-07 |                  -6.5831e-08  |                                       0      |            0 |
| gpt2         |        124439808 | fp32    |          256 | scale_plus_zero_point |                   4   |                  474.7    |              0       |                        474.7    |                    474.7   |               -3.125e-07 |                  -6.5831e-08  |                                       0      |            0 |
| gpt2         |        124439808 | fp16    |           32 | scale_plus_zero_point |                   2   |                  237.35   |              0       |                        237.35   |                    474.7   |              237.35      |                  50           |                                       0      |            0 |
| gpt2         |        124439808 | fp16    |           64 | scale_plus_zero_point |                   2   |                  237.35   |              0       |                        237.35   |                    474.7   |              237.35      |                  50           |                                       0      |            0 |
| gpt2         |        124439808 | fp16    |          128 | scale_plus_zero_point |                   2   |                  237.35   |              0       |                        237.35   |                    474.7   |              237.35      |                  50           |                                       0      |            0 |
| gpt2         |        124439808 | fp16    |          256 | scale_plus_zero_point |                   2   |                  237.35   |              0       |                        237.35   |                    474.7   |              237.35      |                  50           |                                       0      |            0 |
| gpt2         |        124439808 | int8    |           32 | scale_plus_zero_point |                   1   |                  118.675  |             14.8344  |                        133.509  |                    474.7   |              341.191     |                  71.875       |                                      12.5    |      3888744 |
| gpt2         |        124439808 | int8    |           64 | scale_plus_zero_point |                   1   |                  118.675  |              7.41719 |                        126.092  |                    474.7   |              348.608     |                  73.4375      |                                       6.25   |      1944372 |
| gpt2         |        124439808 | int8    |          128 | scale_plus_zero_point |                   1   |                  118.675  |              3.7086  |                        122.384  |                    474.7   |              352.317     |                  74.2187      |                                       3.125  |       972186 |
| gpt2         |        124439808 | int8    |          256 | scale_plus_zero_point |                   1   |                  118.675  |              1.8543  |                        120.529  |                    474.7   |              354.171     |                  74.6094      |                                       1.5625 |       486093 |
| gpt2         |        124439808 | int4    |           32 | scale_plus_zero_point |                   0.5 |                   59.3375 |             14.8344  |                         74.1719 |                    474.7   |              400.528     |                  84.375       |                                      25      |      3888744 |
| gpt2         |        124439808 | int4    |           64 | scale_plus_zero_point |                   0.5 |                   59.3375 |              7.41719 |                         66.7547 |                    474.7   |              407.945     |                  85.9375      |                                      12.5    |      1944372 |
| gpt2         |        124439808 | int4    |          128 | scale_plus_zero_point |                   0.5 |                   59.3375 |              3.7086  |                         63.0461 |                    474.7   |              411.654     |                  86.7187      |                                       6.25   |       972186 |
| gpt2         |        124439808 | int4    |          256 | scale_plus_zero_point |                   0.5 |                   59.3375 |              1.8543  |                         61.1918 |                    474.7   |              413.508     |                  87.1094      |                                       3.125  |       486093 |

## 6. Group-size Sensitivity

| model_name   | dtype   |   group_size |   raw_parameter_memory_MB |   metadata_memory_MB |   effective_parameter_memory_MB |   effective_reduction_percent |   metadata_overhead_percent_of_raw_quantized |
|:-------------|:--------|-------------:|--------------------------:|---------------------:|--------------------------------:|------------------------------:|---------------------------------------------:|
| distilgpt2   | int8    |           32 |                   78.1179 |              9.76474 |                         87.8827 |                       71.875  |                                      12.5    |
| distilgpt2   | int8    |           64 |                   78.1179 |              4.88237 |                         83.0003 |                       73.4375 |                                       6.25   |
| distilgpt2   | int8    |          128 |                   78.1179 |              2.44118 |                         80.5591 |                       74.2188 |                                       3.125  |
| distilgpt2   | int8    |          256 |                   78.1179 |              1.22059 |                         79.3385 |                       74.6094 |                                       1.5625 |
| distilgpt2   | int4    |           32 |                   39.059  |              9.76474 |                         48.8237 |                       84.375  |                                      25      |
| distilgpt2   | int4    |           64 |                   39.059  |              4.88237 |                         43.9413 |                       85.9375 |                                      12.5    |
| distilgpt2   | int4    |          128 |                   39.059  |              2.44118 |                         41.5001 |                       86.7188 |                                       6.25   |
| distilgpt2   | int4    |          256 |                   39.059  |              1.22059 |                         40.2796 |                       87.1094 |                                       3.125  |
| gpt2         | int8    |           32 |                  118.675  |             14.8344  |                        133.509  |                       71.875  |                                      12.5    |
| gpt2         | int8    |           64 |                  118.675  |              7.41719 |                        126.092  |                       73.4375 |                                       6.25   |
| gpt2         | int8    |          128 |                  118.675  |              3.7086  |                        122.384  |                       74.2187 |                                       3.125  |
| gpt2         | int8    |          256 |                  118.675  |              1.8543  |                        120.529  |                       74.6094 |                                       1.5625 |
| gpt2         | int4    |           32 |                   59.3375 |             14.8344  |                         74.1719 |                       84.375  |                                      25      |
| gpt2         | int4    |           64 |                   59.3375 |              7.41719 |                         66.7547 |                       85.9375 |                                      12.5    |
| gpt2         | int4    |          128 |                   59.3375 |              3.7086  |                         63.0461 |                       86.7187 |                                       6.25   |
| gpt2         | int4    |          256 |                   59.3375 |              1.8543  |                         61.1918 |                       87.1094 |                                       3.125  |

## 7. Effective Memory Summary at Group Size 128

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

## 8. Key Findings

|   finding_id | finding                                                                           | evidence                                                                                                      | why_it_matters                                                                                   |
|-------------:|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
|            1 | Metadata overhead slightly reduces theoretical quantization savings.              | With group size 128, int8 effective reduction is about 74.22% and int4 effective reduction is about 86.72%.   | Realistic quantization estimation must include scale/zero-point storage.                         |
|            2 | Metadata overhead affects int4 more than int8 proportionally.                     | Average metadata overhead is about 3.12% of raw int8 parameter memory and 6.25% of raw int4 parameter memory. | Lower-bit formats have smaller raw weight memory, so metadata becomes relatively more important. |
|            3 | Larger group sizes reduce metadata overhead.                                      | Group size 256 stores fewer scale/zero-point groups than group size 32.                                       | There is a tradeoff between quantization granularity and metadata memory.                        |
|            4 | This simulation is still analytical, not real runtime quantized execution.        | The simulation adds metadata assumptions but does not load or run int8/int4 kernels.                          | It prevents overclaiming measured runtime savings.                                               |
|            5 | Quantization mainly reduces parameter memory, not necessarily all runtime memory. | Activations, KV cache, temporary buffers, and allocator behavior may remain unchanged.                        | A full memory estimator must separate parameter memory from total GPU memory.                    |

## 9. Interpretation

Adding metadata overhead makes the quantization estimate more realistic than simple bytes-per-parameter calculation.

int8 and int4 still provide strong parameter-memory reductions, but the effective reduction is slightly lower after metadata is included.

Metadata overhead is proportionally more important for int4 because raw int4 parameter memory is smaller.

Group size matters: smaller group sizes create more groups and therefore more metadata overhead.

## 10. Why This Still Is Not Runtime Measurement

Real quantized execution depends on library implementation, kernel support, activation dtype, KV cache dtype, temporary buffers, memory alignment, and CUDA allocator behavior.

This report only estimates parameter-memory and metadata-memory effects.

## 11. Next Step

The next step is to build a reusable QuantizationMemoryEstimator module in `src/estimators/`.

That module should expose functions to estimate raw parameter memory, metadata overhead, effective parameter memory, and reduction percentage for fp32, fp16, int8, and int4.

## 12. Conclusion

Day 43 shows that quantization-memory estimation should include metadata overhead, especially for low-bit formats like int4.

This makes the quantization extension more realistic and better aligned with large-model memory prediction.