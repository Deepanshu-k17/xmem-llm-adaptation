# Architecture Comparison Setup Report

## 1. Goal

This report starts the architecture comparison phase of the xMem-inspired LLM memory profiling project.

The goal is to compare memory behavior between a CNN-style workload and a Transformer-style workload.

## 2. Scope

This is a controlled small-model comparison.

The goal is not to train large CNNs or large vision foundation models today.

Instead, the goal is to identify architecture-specific memory drivers.

## 3. Model Specifications

| architecture   | model_name                  | input_type     | example_input_shape     | main_layers                                                      | main_memory_drivers                                                             | attention_based   |
|:---------------|:----------------------------|:---------------|:------------------------|:-----------------------------------------------------------------|:--------------------------------------------------------------------------------|:------------------|
| CNN            | SimpleCNN                   | image          | batch x 3 x 224 x 224   | Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear               | image resolution, channels, feature maps, batch size                            | False             |
| Transformer    | SimpleTransformerClassifier | token sequence | batch x sequence_length | Embedding, positional embedding, TransformerEncoderLayer, Linear | sequence length, embedding size, attention heads, feed-forward size, batch size | True              |

## 4. Parameter Memory Comparison

| architecture   | model_name                  |   num_parameters |   fp32_parameter_memory_MB |   fp16_parameter_memory_MB | main_interpretation                                                                                              |
|:---------------|:----------------------------|-----------------:|---------------------------:|---------------------------:|:-----------------------------------------------------------------------------------------------------------------|
| CNN            | SimpleCNN                   |            94538 |                   0.360634 |                   0.180317 | Small CNN has relatively low parameter memory; activation memory can become important for high image resolution. |
| Transformer    | SimpleTransformerClassifier |         11040778 |                  42.1172   |                  21.0586   | Transformer has higher parameter memory due to embeddings, attention layers, and feed-forward blocks.            |

## 5. Forward Memory Profile

| architecture   | model_name   | batch_size   | input_shape        | sequence_length   | peak_allocated_MB   | peak_reserved_MB   | final_allocated_MB   | final_reserved_MB   | scope_note                                            |
|:---------------|:-------------|:-------------|:-------------------|:------------------|:--------------------|:-------------------|:---------------------|:--------------------|:------------------------------------------------------|
| N/A            | N/A          |              | CUDA not available |                   |                     |                    |                      |                     | CUDA not available; forward memory profiling skipped. |

## 6. Key Findings

|   finding_id | finding                                                                      | evidence                                                                                                                                                                                         | why_it_matters                                                                                                |
|-------------:|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
|            1 | CNN and Transformer architectures have different memory drivers.             | CNN memory depends heavily on image resolution, channels, feature maps, and batch size; Transformer memory depends on sequence length, embedding size, attention heads, and feed-forward blocks. | A single memory estimator may not generalize across architectures without architecture-aware features.        |
|            2 | The simple Transformer has much higher parameter memory than the simple CNN. | The Transformer fp32 parameter memory is about 116.79x the CNN fp32 parameter memory.                                                                                                            | Embedding layers and Transformer blocks can dominate parameter memory even in small Transformer-style models. |
|            3 | Attention-based models require sequence-length-aware memory estimation.      | Transformer memory depends on token sequence length and attention computation.                                                                                                                   | This supports the need for separate handling of sequence length in LLM memory estimation.                     |
|            4 | CNN-style workloads need image-resolution and feature-map-aware estimation.  | CNN activations are spatial tensors whose size depends on height, width, channels, and batch size.                                                                                               | Vision-style models require different memory features than text Transformer models.                           |
|            5 | Day 56 is a setup day for architecture comparison.                           | Simple CNN and Transformer models were defined, parameter memory was compared, and optional forward profiling was added.                                                                         | This creates the foundation for deeper architecture comparison in the next days.                              |

## 7. Main Interpretation

CNN-style and Transformer-style models have different memory behavior.

CNN memory depends strongly on spatial dimensions, feature maps, channels, and batch size.

Transformer memory depends strongly on sequence length, embedding size, attention heads, feed-forward size, and batch size.

This means memory estimation should be architecture-aware.

## 8. Limitations

- Only simple toy architectures are used today.
- No full-scale vision foundation model is profiled.
- No training memory comparison is performed today.
- CNN activation memory is not deeply decomposed yet.
- Transformer attention memory is not deeply decomposed yet.

## 9. Next Step

Day 57 will analyze how CNN memory scales with image resolution and batch size.

## 10. Conclusion

Day 56 sets up the architecture comparison phase by defining simple CNN and Transformer baselines and comparing their parameter-memory behavior.