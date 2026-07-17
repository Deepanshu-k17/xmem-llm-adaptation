# CNN Memory Scaling Report

## 1. Goal

This report analyzes how CNN memory changes with image resolution and batch size.

The goal is to understand CNN-side memory behavior before comparing it with Transformer-style memory behavior.

## 2. Model

The model used is `SimpleCNN`, a small convolutional classifier with three convolution layers and one linear classifier.

- Parameter count: 94538
- fp32 parameter memory: 0.3606 MB

## 3. Scope

This day focuses only on CNN memory scaling.

Transformer scaling will be analyzed separately.

## 4. Analytical Activation-memory Estimate

|   batch_size |   image_size |   estimated_total_activation_memory_MB | scope_note                                                               |
|-------------:|-------------:|---------------------------------------:|:-------------------------------------------------------------------------|
|            1 |           64 |                                1.10986 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            1 |          128 |                                4.43799 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            1 |          224 |                               13.5903  | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            1 |          384 |                               39.938   | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            2 |           64 |                                2.21973 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            2 |          128 |                                8.87598 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            2 |          224 |                               27.1807  | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            2 |          384 |                               79.876   | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            4 |           64 |                                4.43945 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            4 |          128 |                               17.752   | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            4 |          224 |                               54.3613  | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            4 |          384 |                              159.752   | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            8 |           64 |                                8.87891 | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            8 |          128 |                               35.5039  | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            8 |          224 |                              108.723   | Analytical activation-memory estimate for SimpleCNN forward activations. |
|            8 |          384 |                              319.504   | Analytical activation-memory estimate for SimpleCNN forward activations. |

## 5. Forward Memory Profile

| architecture   | model_name   |   batch_size |   image_size | input_shape       |   num_parameters |   fp32_parameter_memory_MB |   analytical_activation_memory_MB |   peak_allocated_MB |   peak_reserved_MB |   final_allocated_MB |   final_reserved_MB | profiling_status   | scope_note                                       |
|:---------------|:-------------|-------------:|-------------:|:------------------|-----------------:|---------------------------:|----------------------------------:|--------------------:|-------------------:|---------------------:|--------------------:|:-------------------|:-------------------------------------------------|
| CNN            | SimpleCNN    |            1 |           64 | 1 x 3 x 64 x 64   |            94538 |                   0.360634 |                           1.10986 |             9.53467 |                 22 |              9.53418 |                  22 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            2 |           64 | 2 x 3 x 64 x 64   |            94538 |                   0.360634 |                           2.21973 |            11.5806  |                 24 |              9.58105 |                  24 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            4 |           64 | 4 x 3 x 64 x 64   |            94538 |                   0.360634 |                           4.43945 |            13.6743  |                 26 |              9.6748  |                  26 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            8 |           64 | 8 x 3 x 64 x 64   |            94538 |                   0.360634 |                           8.87891 |            17.8618  |                 26 |              9.8623  |                  26 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            1 |          128 | 1 x 3 x 128 x 128 |            94538 |                   0.360634 |                           4.43799 |            13.6743  |                 26 |              9.6748  |                  26 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            2 |          128 | 2 x 3 x 128 x 128 |            94538 |                   0.360634 |                           8.87598 |            17.8618  |                 26 |              9.8623  |                  26 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            4 |          128 | 4 x 3 x 128 x 128 |            94538 |                   0.360634 |                          17.752   |            26.2368  |                 46 |             10.2373  |                  46 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            8 |          128 | 8 x 3 x 128 x 128 |            94538 |                   0.360634 |                          35.5039  |            42.9868  |                 56 |             10.9873  |                  56 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            1 |          224 | 1 x 3 x 224 x 224 |            94538 |                   0.360634 |                          13.5903  |            22.311   |                 44 |             10.0615  |                  44 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            2 |          224 | 2 x 3 x 224 x 224 |            94538 |                   0.360634 |                          27.1807  |            35.1353  |                 52 |             10.6357  |                  52 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            4 |          224 | 4 x 3 x 224 x 224 |            94538 |                   0.360634 |                          54.3613  |            60.7837  |                 76 |             11.7842  |                  76 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            8 |          224 | 8 x 3 x 224 x 224 |            94538 |                   0.360634 |                         108.723   |           114.081   |                124 |             14.0811  |                 124 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            1 |          384 | 1 x 3 x 384 x 384 |            94538 |                   0.360634 |                          39.938   |            47.1743  |                 60 |             11.1748  |                  60 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            2 |          384 | 2 x 3 x 384 x 384 |            94538 |                   0.360634 |                          79.876   |            84.8618  |                 96 |             12.8623  |                  96 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            4 |          384 | 4 x 3 x 384 x 384 |            94538 |                   0.360634 |                         159.752   |           160.237   |                168 |             16.2373  |                 168 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |
| CNN            | SimpleCNN    |            8 |          384 | 8 x 3 x 384 x 384 |            94538 |                   0.360634 |                         319.504   |           311.487   |                326 |             23.4873  |                 326 | measured_cuda      | Measured CUDA forward-pass memory for SimpleCNN. |

## 6. Scaling Summary

| scaling_dimension   | fixed_value    |   min_batch_size |   max_batch_size |   activation_memory_at_min_batch_MB |   activation_memory_at_max_batch_MB | scaling_interpretation                                                                                    |   min_image_size |   max_image_size |   activation_memory_at_min_resolution_MB |   activation_memory_at_max_resolution_MB |
|:--------------------|:---------------|-----------------:|-----------------:|------------------------------------:|------------------------------------:|:----------------------------------------------------------------------------------------------------------|-----------------:|-----------------:|-----------------------------------------:|-----------------------------------------:|
| batch_size          | image_size=64  |                1 |                8 |                             1.10986 |                             8.87891 | CNN activation memory scales approximately linearly with batch size.                                      |              nan |              nan |                                nan       |                                  nan     |
| batch_size          | image_size=128 |                1 |                8 |                             4.43799 |                            35.5039  | CNN activation memory scales approximately linearly with batch size.                                      |              nan |              nan |                                nan       |                                  nan     |
| batch_size          | image_size=224 |                1 |                8 |                            13.5903  |                           108.723   | CNN activation memory scales approximately linearly with batch size.                                      |              nan |              nan |                                nan       |                                  nan     |
| batch_size          | image_size=384 |                1 |                8 |                            39.938   |                           319.504   | CNN activation memory scales approximately linearly with batch size.                                      |              nan |              nan |                                nan       |                                  nan     |
| image_resolution    | batch_size=1   |              nan |              nan |                           nan       |                           nan       | CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W. |               64 |              384 |                                  1.10986 |                                   39.938 |
| image_resolution    | batch_size=2   |              nan |              nan |                           nan       |                           nan       | CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W. |               64 |              384 |                                  2.21973 |                                   79.876 |
| image_resolution    | batch_size=4   |              nan |              nan |                           nan       |                           nan       | CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W. |               64 |              384 |                                  4.43945 |                                  159.752 |
| image_resolution    | batch_size=8   |              nan |              nan |                           nan       |                           nan       | CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W. |               64 |              384 |                                  8.87891 |                                  319.504 |

## 7. Key Findings

|   finding_id | finding                                                                          | evidence                                                                                                         | why_it_matters                                                                                      |
|-------------:|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
|            1 | CNN activation memory scales approximately linearly with batch size.             | At image size 224, increasing batch size from 1 to 8 increased analytical activation memory by 8.00x.            | Batch size is a direct memory-scaling factor for CNN workloads.                                     |
|            2 | CNN activation memory grows strongly with image resolution.                      | At batch size 1, increasing image size from 64 to 384 increased analytical activation memory by 35.98x.          | CNN feature maps depend on spatial dimensions, so higher image resolution increases memory sharply. |
|            3 | CNN parameter memory is small compared with activation memory for larger inputs. | SimpleCNN fp32 parameter memory is only 0.3606 MB, while activation memory grows with image size and batch size. | For CNN inference, activations can dominate memory even when parameter count is small.              |
|            4 | CNN memory estimation needs image-resolution-aware features.                     | Activation memory changes with H x W feature-map dimensions.                                                     | A GPT-style sequence-length estimator is not enough for CNN-style workloads.                        |
|            5 | Day 57 focuses on CNN-side scaling only.                                         | Transformer scaling will be analyzed separately in the next architecture-comparison step.                        | Separating CNN and Transformer scaling makes the comparison cleaner.                                |

## 8. Main Interpretation

CNN activation memory depends heavily on spatial feature-map size.

The feature-map tensors scale with batch size, channel count, height, and width.

Therefore, CNN memory grows approximately linearly with batch size and strongly with image resolution.

Even though the SimpleCNN parameter memory is very small, activation memory can grow quickly for larger images and larger batches.

## 9. Limitations

- This uses a small CNN, not a production-scale vision model.
- Analytical activation memory is simplified.
- If CUDA is unavailable, runtime peak memory is not measured.
- Backward/training memory is not included today.
- cuDNN/workspace memory is not analytically modeled.

## 10. Next Step

Day 58 will analyze Transformer memory scaling with sequence length and batch size.

## 11. Conclusion

Day 57 shows that CNN memory estimation must include image-resolution and feature-map-aware terms.