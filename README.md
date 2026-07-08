# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference and training workloads.

The goal is to study how peak GPU memory changes with model size, sequence length, generation length, batch size, dtype, optimizer choice, CUDA allocator behavior, quantization assumptions, sparsity assumptions, and model-parallel partitioning assumptions.

The project builds estimator modules that predict or approximate:

- inference peak allocated memory
- inference peak reserved memory
- training peak allocated memory
- training peak reserved memory
- optimizer-state memory
- precision/quantization memory changes
- sparsity memory changes
- model-parallel per-device memory

---

## Current Status

The project currently has completed work in five main phases:

1. **Inference memory profiling and prediction**
2. **Training memory profiling and prediction**
3. **Precision and quantization memory adaptation**
4. **Sparsity memory estimation**
5. **Model-parallel memory partitioning simulation**

The project has been tested or analytically simulated on:

- `sshleifer/tiny-gpt2`
- `distilgpt2`
- `gpt2`

Important clarification:

```text
Quantization int8/int4, sparsity, and model-parallel results are analytical memory estimates, not measured runtime execution results.
```

---

## Models Used

| Model                 | Purpose                                    |
| --------------------- | ------------------------------------------ |
| `sshleifer/tiny-gpt2` | Pipeline validation                        |
| `distilgpt2`          | Main inference/training memory experiments |
| `gpt2`                | Larger GPT-style validation model          |

---

## What Was Profiled

The profiling pipeline records:

- model name
- GPU name
- batch size
- input tokens
- generated tokens
- dtype
- cache setting
- optimizer name
- peak allocated memory
- peak reserved memory
- final allocated memory
- final reserved memory
- runtime
- OOM status
- error message

For training workloads, stage-level memory is also logged:

- before model load
- after model load
- after batch creation
- after optimizer creation
- after forward pass
- after loss computation
- after backward pass
- after optimizer step
- after zero_grad

Stage-level logging is important because peak training memory can occur temporarily during backward pass or optimizer step.

---

## Estimator Modules

The project currently includes:

```text
src/estimators/
    model_config_utils.py
    base_estimator.py
    improved_base_estimator.py
    kv_cache_estimator.py
    precision_estimator.py
    allocator_correction.py
    combined_inference_estimator.py
    optimizer_state_estimator.py
    training_memory_estimator.py
    training_memory_estimator_v4.py
    quantization_memory_estimator.py
    sparsity_memory_estimator.py
    model_parallel_memory_estimator.py
```

### Main estimator roles

| Estimator                      | Purpose                                                   |
| ------------------------------ | --------------------------------------------------------- |
| `CombinedInferenceEstimator`   | Predicts inference peak allocated/reserved memory         |
| `OptimizerStateEstimator`      | Estimates optimizer-state memory, especially AdamW vs SGD |
| `TrainingMemoryEstimatorV4`    | Final training-memory estimator                           |
| `QuantizationMemoryEstimator`  | Analytical quantization parameter-memory estimator        |
| `SparsityMemoryEstimator`      | Analytical sparse parameter-memory estimator              |
| `ModelParallelMemoryEstimator` | Analytical per-device model-parallel memory estimator     |

---

# Phase 1: Inference Memory Prediction

## Key Inference Results

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

| Metric                    | Value |
| ------------------------- | ----: |
| peak allocated mean error | 2.36% |
| peak reserved mean error  | 2.36% |

For `gpt2`, the estimator achieved:

| Metric                    | Value |
| ------------------------- | ----: |
| peak allocated mean error | 2.50% |
| peak reserved mean error  | 2.28% |
| allocated MRE             | 2.66% |
| reserved MRE              | 2.06% |

This supports that the inference estimator generalizes beyond `distilgpt2`.

---

## Inference PEF-style Fit/Fail Simulation

Across 720 simulated fit/fail cases:

| Metric                    |  Value |
| ------------------------- | -----: |
| overall accuracy          | 94.17% |
| dangerous failure rate    |  5.56% |
| conservative failure rate |  0.28% |

For `distilgpt2`:

| Metric                 |  Value |
| ---------------------- | -----: |
| accuracy               | 99.52% |
| dangerous failure rate |  0.00% |

---

# Phase 2: Training Memory Prediction

## Training vs Inference

For `distilgpt2`, training used around **4.7x to 4.8x** more peak allocated memory than inference under comparable settings.

Training memory needs separate modeling because it includes:

- gradients
- optimizer states
- backward temporary tensors
- optimizer-step memory behavior

---

## Optimizer Comparison

For `distilgpt2` training with batch size 1 and input length 64:

| Optimizer | Peak allocated memory | Peak reserved memory | Final allocated memory |
| --------- | --------------------: | -------------------: | ---------------------: |
| SGD       |             956.49 MB |            1030.0 MB |              345.63 MB |
| AdamW     |            1600.94 MB |            1738.0 MB |              973.48 MB |

AdamW used **644.45 MB** more peak allocated memory than SGD.

For `gpt2`, AdamW current allocated memory increased from **983.20 MB** after backward to **1934.12 MB** after optimizer step, a jump of about **950.92 MB**.

This matches the expected AdamW optimizer-state behavior.

---

## OptimizerStateEstimator

The `OptimizerStateEstimator` models optimizer-state memory using optimizer-specific factors:

| Optimizer         | Optimizer-state factor |
| ----------------- | ---------------------: |
| SGD               |   0 × parameter memory |
| SGD with momentum |   1 × parameter memory |
| Adam              |   2 × parameter memory |
| AdamW             |   2 × parameter memory |

For `distilgpt2` in fp32:

| Metric                                       |     Value |
| -------------------------------------------- | --------: |
| parameter memory                             | 312.47 MB |
| estimated AdamW optimizer-state memory       | 624.94 MB |
| observed AdamW-SGD peak allocated difference | 644.45 MB |
| relative error                               |     3.03% |

For `gpt2` in fp32:

| Metric                                 |     Value |
| -------------------------------------- | --------: |
| parameter memory                       | 474.70 MB |
| estimated AdamW optimizer-state memory | 949.40 MB |
| observed AdamW optimizer-step jump     | 950.92 MB |

This supports modeling AdamW optimizer state as approximately **2 × parameter memory**.

---

## TrainingMemoryEstimator V4

TrainingMemoryEstimator V4 uses optimizer-specific and model-size-aware backward temporary memory correction.

On the combined `distilgpt2` + `gpt2` training validation set, V4 achieved:

| Metric               | Value |
| -------------------- | ----: |
| allocated MRE        | 2.98% |
| allocated mean error | 3.66% |
| allocated max error  | 8.77% |
| reserved MRE         | 3.69% |
| reserved mean error  | 4.29% |
| reserved max error   | 9.52% |

V4 is the current final training estimator.

---

## V2 vs V3 vs V4

| Estimator | Allocated MRE | Allocated mean error | Allocated max error | Reserved MRE | Reserved mean error | Reserved max error |
| --------- | ------------: | -------------------: | ------------------: | -----------: | ------------------: | -----------------: |
| V2        |         2.99% |                4.95% |              17.69% |        3.69% |               5.00% |             16.83% |
| V3        |         2.99% |                4.36% |              12.77% |        4.20% |               4.99% |             12.52% |
| V4        |         2.98% |                3.66% |               8.77% |        3.69% |               4.29% |              9.52% |

V4 reduced worst-case error while preserving good performance across both `distilgpt2` and `gpt2`.

---

## V4 Training PEF-style Fit/Fail Simulation

Across **266 simulated training placement cases**, V4 achieved:

| Metric                    |  Value |
| ------------------------- | -----: |
| total cases               |    266 |
| correct cases             |    256 |
| dangerous failures        |      8 |
| conservative failures     |      2 |
| accuracy                  | 96.24% |
| dangerous failure rate    |  3.01% |
| conservative failure rate |  0.75% |

With a **7.5% safety margin**:

| Metric                |  Value |
| --------------------- | -----: |
| accuracy              | 95.11% |
| dangerous failures    |      0 |
| conservative failures |     13 |

Recommended safe placement estimator:

```text
TrainingMemoryEstimator V4 + 7.5% safety margin
```

---

# Phase 3: Precision and Quantization Memory Adaptation

This phase studies how dtype and analytical quantization assumptions affect parameter memory and estimated total memory.

Important clarification:

```text
fp32/fp16 results include measured runtime memory.
int8/int4 results are analytical estimates only.
```

---

## fp32 vs fp16 Precision Results

| Model      | fp32 peak allocated | fp16 peak allocated | Reduction |
| ---------- | ------------------: | ------------------: | --------: |
| distilgpt2 |           343.41 MB |           181.25 MB |    47.22% |
| gpt2       |           504.11 MB |           264.78 MB |    47.48% |

Average measured fp16 allocated-memory reduction:

| Metric            |  Value |
| ----------------- | -----: |
| average reduction | 47.35% |
| minimum reduction | 47.22% |
| maximum reduction | 47.48% |

The theoretical fp16 parameter-memory reduction is 50%, but measured runtime memory reduction is lower because runtime memory also includes activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

---

## Analytical Quantization Parameter-memory Estimates

| Dtype | Bits per parameter | Bytes per parameter | Theoretical parameter-memory reduction vs fp32 |
| ----- | -----------------: | ------------------: | ---------------------------------------------: |
| fp32  |                 32 |                 4.0 |                                             0% |
| fp16  |                 16 |                 2.0 |                                            50% |
| int8  |                  8 |                 1.0 |                                            75% |
| int4  |                  4 |                 0.5 |                                          87.5% |

For `distilgpt2`:

| Dtype | Estimated raw parameter memory |
| ----- | -----------------------------: |
| fp32  |                      312.47 MB |
| fp16  |                      156.24 MB |
| int8  |                       78.12 MB |
| int4  |                       39.06 MB |

For `gpt2`:

| Dtype | Estimated raw parameter memory |
| ----- | -----------------------------: |
| fp32  |                      474.70 MB |
| fp16  |                      237.35 MB |
| int8  |                      118.68 MB |
| int4  |                       59.34 MB |

---

## Quantization Metadata Overhead

The simulation assumes grouped quantization metadata:

```text
scale + zero-point metadata = 4 bytes per group
default group size = 128
```

At group size 128:

| Model      | Dtype | Raw parameter memory | Metadata memory | Effective parameter memory | Effective reduction |
| ---------- | ----- | -------------------: | --------------: | -------------------------: | ------------------: |
| distilgpt2 | int8  |             78.12 MB |         2.44 MB |                   80.56 MB |              74.22% |
| distilgpt2 | int4  |             39.06 MB |         2.44 MB |                   41.50 MB |              86.72% |
| gpt2       | int8  |            118.68 MB |         3.71 MB |                  122.38 MB |              74.22% |
| gpt2       | int4  |             59.34 MB |         3.71 MB |                   63.05 MB |              86.72% |

Metadata overhead affects int4 more strongly than int8 because raw int4 memory is smaller.

---

## Quantization-aware Total-memory Approximation

The approximation uses:

```text
estimated_total_memory(dtype)
=
measured_fp32_total_memory
- fp32_parameter_memory
+ effective_parameter_memory(dtype)
```

Equivalent:

```text
estimated_total_memory(dtype)
=
non_parameter_memory
+ effective_parameter_memory(dtype)
```

### fp32 Runtime Memory Decomposition

| Model      | fp32 measured total | fp32 parameter memory | non-parameter memory | parameter fraction | non-parameter fraction |
| ---------- | ------------------: | --------------------: | -------------------: | -----------------: | ---------------------: |
| distilgpt2 |           343.41 MB |             312.47 MB |             30.94 MB |             90.99% |                  9.01% |
| gpt2       |           504.11 MB |             474.70 MB |             29.41 MB |             94.17% |                  5.83% |

### Estimated Total-memory Reduction

For `distilgpt2`:

| Dtype | Estimated total memory | Estimated total reduction | Measured total reduction |
| ----- | ---------------------: | ------------------------: | -----------------------: |
| fp32  |              343.41 MB |                     0.00% |                    0.00% |
| fp16  |              187.17 MB |                    45.50% |                   47.22% |
| int8  |              111.50 MB |                    67.53% |                      N/A |
| int4  |               72.44 MB |                    78.91% |                      N/A |

For `gpt2`:

| Dtype | Estimated total memory | Estimated total reduction | Measured total reduction |
| ----- | ---------------------: | ------------------------: | -----------------------: |
| fp32  |              504.11 MB |                     0.00% |                    0.00% |
| fp16  |              266.76 MB |                    47.08% |                   47.48% |
| int8  |              151.79 MB |                    69.89% |                      N/A |
| int4  |               92.46 MB |                    81.66% |                      N/A |

---

# Phase 4: Sparsity Memory Estimation

This phase estimates how unstructured sparsity may affect parameter memory and total memory.

Important clarification:

```text
Sparsity results are analytical storage estimates only.
No real sparse CUDA kernels or sparse model execution were measured.
```

---

## SparsityMemoryEstimator

The project includes:

```text
src/estimators/sparsity_memory_estimator.py
```

The estimator computes:

- dense fp32 parameter memory
- nonzero parameter count
- sparse value memory
- sparse index metadata memory
- sparse total parameter memory
- memory reduction compared with dense fp32

The current simple unstructured sparse fp32 model assumes:

```text
value bytes = 4
index bytes per nonzero = 4
```

So each nonzero sparse value effectively costs:

```text
value + index = 8 bytes
```

Dense fp32 costs:

```text
4 bytes per parameter
```

This creates an important break-even behavior.

---

## Sparse Parameter-memory Results

| Sparsity | Interpretation                           | Parameter-memory reduction |
| -------: | ---------------------------------------- | -------------------------: |
|       0% | Sparse storage is worse than dense       |                      -100% |
|      25% | Sparse storage is still worse than dense |                       -50% |
|      50% | Break-even point                         |                         0% |
|      75% | Useful sparse memory reduction           |                        50% |
|      90% | Strong sparse memory reduction           |                        80% |

Main insight:

```text
Unstructured sparsity is not automatically memory-efficient because index metadata can cancel out the savings from zero weights.
```

---

## Sparsity-aware Total-memory Approximation

The approximation uses:

```text
estimated_total_memory(sparsity)
=
measured_fp32_total_memory
- dense_fp32_parameter_memory
+ sparse_total_parameter_memory(sparsity)
```

Equivalent:

```text
estimated_total_memory(sparsity)
=
non_parameter_memory
+
sparse_total_parameter_memory(sparsity)
```

### distilgpt2

| Sparsity | Estimated total memory | Reduction vs fp32 |
| -------: | ---------------------: | ----------------: |
|       0% |              655.88 MB |           -90.99% |
|      25% |              499.65 MB |           -45.50% |
|      50% |              343.41 MB |             0.00% |
|      75% |              187.17 MB |            45.50% |
|      90% |               93.43 MB |            72.79% |

### gpt2

| Sparsity | Estimated total memory | Reduction vs fp32 |
| -------: | ---------------------: | ----------------: |
|       0% |              978.81 MB |           -94.17% |
|      25% |              741.46 MB |           -47.08% |
|      50% |              504.11 MB |             0.00% |
|      75% |              266.76 MB |            47.08% |
|      90% |              124.35 MB |            75.33% |

---

## Sparsity vs Quantization

| Method                    | Main memory effect                           | Runtime measured?       | Main tradeoff                                                          |
| ------------------------- | -------------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| fp16                      | Reduces bytes per parameter                  | Yes, fp32/fp16 measured | Practical and widely supported                                         |
| int8                      | Reduces bytes per parameter                  | No                      | Strong analytical saving, needs real quantized execution validation    |
| int4                      | Reduces bytes per parameter                  | No                      | Larger estimated saving, but quality/kernel support matters            |
| 75% unstructured sparsity | Removes zero weights but stores indices      | No                      | Similar total-memory saving to fp16 in this approximation              |
| 90% unstructured sparsity | Removes more zero weights but stores indices | No                      | Strong saving, but real sparse runtime depends on hardware and kernels |

Main sparsity conclusion:

```text
Sparsity is more complicated than quantization because unstructured sparsity requires index metadata.
```

---

# Phase 5: Model-parallel Memory Partitioning Simulation

This phase analytically estimates how splitting model parameters across multiple devices affects per-device memory.

Important clarification:

```text
Model-parallel results are analytical memory-planning estimates only.
No real tensor parallelism, pipeline parallelism, NCCL, or multi-GPU execution was implemented.
```

---

## ModelParallelMemoryEstimator

The project includes:

```text
src/estimators/model_parallel_memory_estimator.py
```

The estimator computes:

- total parameter memory
- ideal partitioned parameter memory
- replicated overhead memory
- communication buffer memory
- total overhead memory
- estimated per-device memory
- ideal memory reduction
- effective memory reduction

Supported dtypes:

```text
fp32
fp16
bf16
```

---

## Model-parallel Assumptions

The current analytical simulation assumes:

```text
replication overhead = 5%
communication buffer overhead = 3%
combined overhead = 8%
```

Ideal partitioned memory:

```text
ideal_partitioned_memory = total_parameter_memory / num_devices
```

Estimated per-device memory:

```text
estimated_per_device_memory
=
ideal_partitioned_memory
+
replicated_overhead_memory
+
communication_buffer_memory
```

---

## Model-parallel Results

With 8% combined overhead:

| Number of devices | Ideal reduction | Effective reduction |
| ----------------: | --------------: | ------------------: |
|                 1 |           0.00% |              -8.00% |
|                 2 |          50.00% |              42.00% |
|                 4 |          75.00% |              67.00% |
|                 8 |          87.50% |              79.50% |

Main insight:

```text
Model parallelism reduces per-device parameter memory, but overhead prevents perfect linear scaling.
```

---

## Example Per-device Memory

### distilgpt2 fp32

| Devices | Estimated per-device memory |
| ------: | --------------------------: |
|       1 |                   337.47 MB |
|       2 |                   181.23 MB |
|       4 |                   103.12 MB |
|       8 |                    64.06 MB |

### gpt2 fp32

| Devices | Estimated per-device memory |
| ------: | --------------------------: |
|       1 |                   512.68 MB |
|       2 |                   275.33 MB |
|       4 |                   156.65 MB |
|       8 |                    97.31 MB |

### gpt2 fp16

| Devices | Estimated per-device memory |
| ------: | --------------------------: |
|       1 |                   256.34 MB |
|       2 |                   137.66 MB |
|       4 |                    78.33 MB |
|       8 |                    48.66 MB |

---

# Final Estimator Recommendations

| Use case                                    | Recommended estimator                             | Reason                                                   |
| ------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| LLM inference memory prediction             | `CombinedInferenceEstimator`                      | Low prediction error on `distilgpt2` and `gpt2`          |
| LLM training raw memory prediction          | `TrainingMemoryEstimator V4`                      | Best combined training estimator                         |
| Safe training placement                     | `TrainingMemoryEstimator V4 + 7.5% safety margin` | Removes dangerous underpredictions in current validation |
| Quantization parameter-memory estimation    | `QuantizationMemoryEstimator`                     | Includes dtype size and metadata overhead                |
| Sparsity parameter-memory estimation        | `SparsityMemoryEstimator`                         | Separates value memory and index metadata                |
| Model-parallel per-device memory estimation | `ModelParallelMemoryEstimator`                    | Separates ideal partitioning and overhead                |

---

# Project Structure

```text
src/
    experiment_logger.py
    training_logger.py
    estimators/
        model_config_utils.py
        base_estimator.py
        improved_base_estimator.py
        kv_cache_estimator.py
        precision_estimator.py
        allocator_correction.py
        combined_inference_estimator.py
        optimizer_state_estimator.py
        training_memory_estimator.py
        training_memory_estimator_v4.py
        quantization_memory_estimator.py
        sparsity_memory_estimator.py
        model_parallel_memory_estimator.py

results/
    inference_clean.csv
    inference_summary.csv
    estimator_comparison.csv
    reserved_memory_summary.csv
    key_findings.csv

    gpt2_inference_validation.csv
    gpt2_inference_estimator_predictions.csv
    gpt2_inference_estimator_metrics.csv

    training_estimator_v4_predictions.csv
    training_estimator_v4_metrics.csv
    training_estimator_v4_error_by_model.csv
    training_estimator_v4_error_by_optimizer.csv
    training_estimator_v4_error_by_sequence_length.csv
    training_estimator_v2_v3_v4_metrics.csv

    v4_training_pef_cases.csv
    v4_training_pef_summary.csv
    v4_training_pef_by_memory_limit.csv
    v4_training_pef_by_model.csv
    v4_training_pef_by_optimizer.csv

    precision_memory_comparison.csv
    precision_memory_key_findings.csv
    precision_reduction_summary.csv

    quantization_memory_formulas.csv
    quantization_parameter_memory_simulation.csv
    quantization_memory_reduction_summary.csv
    quantization_key_findings.csv
    quantization_metadata_overhead_simulation.csv
    quantization_group_size_sensitivity.csv
    quantization_effective_memory_summary.csv
    quantization_metadata_key_findings.csv
    quantization_estimator_demo.csv
    quantization_estimator_summary.csv
    quantization_estimator_key_findings.csv
    quantization_dtype_comparison.csv
    quantization_measured_vs_analytical.csv
    quantization_dtype_key_findings.csv
    quantization_total_memory_approximation.csv
    quantization_memory_decomposition.csv
    quantization_total_memory_key_findings.csv
    precision_quantization_phase_summary.csv

    sparsity_memory_formulas.csv
    sparsity_parameter_memory_simulation.csv
    sparsity_memory_reduction_summary.csv
    sparsity_key_findings.csv
    sparsity_estimator_demo.csv
    sparsity_estimator_summary.csv
    sparsity_estimator_key_findings.csv
    sparsity_total_memory_approximation.csv
    sparsity_memory_decomposition.csv
    sparsity_total_memory_key_findings.csv
    sparsity_phase_summary.csv
    sparsity_vs_quantization_comparison.csv
    sparsity_final_findings.csv
    sparsity_recommendations.csv

    model_parallel_memory_formulas.csv
    model_parallel_parameter_partitioning.csv
    model_parallel_key_findings.csv
    model_parallel_estimator_demo.csv
    model_parallel_estimator_summary.csv
    model_parallel_estimator_key_findings.csv

plots/
    actual_vs_predicted_allocated.png
    actual_vs_predicted_reserved.png
    pef_failure_rate_by_limit.png
    pef_accuracy_by_model.png
    dangerous_failure_by_model.png

    precision_peak_allocated_comparison.png
    precision_peak_reserved_comparison.png
    precision_memory_reduction_percent.png

    quantization_parameter_memory_by_dtype.png
    quantization_memory_reduction_percent.png
    quantization_effective_parameter_memory.png
    quantization_total_memory_by_dtype.png
    quantization_total_memory_reduction.png

    sparsity_parameter_memory_by_level.png
    sparsity_memory_reduction_percent.png
    sparsity_total_memory_by_level.png
    sparsity_parameter_vs_non_parameter.png
    sparsity_total_memory_reduction.png

    model_parallel_parameter_memory_by_devices_fp32.png
    model_parallel_parameter_memory_by_devices_fp16.png
    model_parallel_memory_reduction_percent_fp32.png
    model_parallel_memory_reduction_percent_fp16.png
    model_parallel_replication_overhead.png

report/
    inference_phase_report.md
    pef_style_simulation.md
    training_phase_report.md
    optimizer_state_estimator_report.md
    training_memory_estimator_report.md
    training_estimator_improvement_report.md
    training_estimator_v4_model_size_correction_report.md
    final_training_estimator_report.md
    v4_training_pef_simulation_report.md
    v4_training_pef_safety_margin_report.md
    final_estimator_comparison_report.md

    precision_memory_adaptation_report.md
    quantization_memory_theory_report.md
    quantization_metadata_overhead_report.md
    quantization_memory_estimator_report.md
    quantization_dtype_comparison_report.md
    quantization_total_memory_approximation_report.md
    precision_quantization_phase_summary_report.md

    sparsity_memory_theory_report.md
    sparsity_memory_estimator_report.md
    sparsity_total_memory_approximation_report.md
    sparsity_phase_summary_report.md

    model_parallel_memory_theory_report.md
    model_parallel_memory_estimator_report.md
```

---

# Main Interpretation

This project shows that LLM memory behavior cannot be explained by parameter count alone.

Inference memory is affected by:

- model parameters
- batch size
- sequence length
- generation length
- KV cache
- dtype
- framework overhead
- CUDA allocator behavior

Training memory is affected by additional components:

- gradients
- optimizer states
- backward temporary tensors
- optimizer-step behavior

The training estimator required separate modeling from the inference estimator.

AdamW memory behavior was explained using optimizer-state estimation, where AdamW stores approximately **2 × parameter memory** as optimizer state.

TrainingMemoryEstimator V4 is the strongest current training estimator because it combines optimizer-specific and model-size-aware correction.

The optimization-aware phase shows:

- fp16 gives strong measured memory reduction
- int8/int4 quantization can give larger analytical memory reductions
- sparsity is useful only when sparsity is high enough to overcome metadata overhead
- model parallelism reduces per-device memory but overhead prevents perfect scaling

---

# Limitations

Current limitations:

- Experiments are limited to single-GPU Colab/T4-style environments.
- The main validation models are `distilgpt2` and `gpt2`.
- Larger LLMs still need validation.
- Reserved memory can vary across CUDA/PyTorch runtime environments.
- int8/int4 quantization results are analytical estimates only.
- No real int8/int4 quantized model execution has been profiled yet.
- Sparsity results are analytical storage estimates only.
- No sparse CUDA kernels or sparse runtime execution were profiled.
- Model-parallel results are analytical memory-planning estimates only.
- No real tensor parallelism, pipeline parallelism, NCCL, or multi-GPU execution was implemented.
- Activation partitioning is not modeled.
- Optimizer-state partitioning under model parallelism is not modeled.
- Latency, throughput, and quality impact are not evaluated.

---

# Next Work

Planned next steps:

1. Build model-parallel total-memory approximation.
2. Compare model parallelism with quantization and sparsity.
3. Add architecture comparison between CNN-style and Transformer-style workloads.
4. Prepare final combined project report.
5. Update final project README after all phases are complete.
6. Prepare final mentor/interview explanation document.

---

# Final Current Status

The project currently has strong completed work in:

- LLM inference memory profiling
- LLM inference memory prediction
- gpt2 inference validation
- LLM training memory profiling
- optimizer-state estimation
- training memory prediction
- gpt2 training validation
- TrainingMemoryEstimator V4
- V4 training PEF-style fit/fail simulation
- V4 safety-margin analysis
- fp32/fp16 precision-memory analysis
- analytical quantization-memory estimation
- quantization metadata overhead simulation
- reusable `QuantizationMemoryEstimator`
- quantization-aware total-memory approximation
- analytical sparsity memory estimation
- reusable `SparsityMemoryEstimator`
- sparsity-aware total-memory approximation
- sparsity vs quantization comparison
- analytical model-parallel memory partitioning
- reusable `ModelParallelMemoryEstimator`

The next phase is model-parallel total-memory approximation and comparison with other memory-optimization strategies.
