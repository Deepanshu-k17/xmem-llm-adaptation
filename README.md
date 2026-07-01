# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference and training workloads.

The goal is to study how peak GPU memory changes with model size, sequence length, generation length, batch size, dtype, optimizer choice, CUDA allocator behavior, and quantization-related memory assumptions, then build estimator modules that predict peak allocated and peak reserved GPU memory.

The project currently covers:

* inference memory profiling
* inference memory estimation
* gpt2 inference validation
* PEF-style fit/fail simulation for inference
* training memory profiling
* optimizer-state memory estimation
* training memory estimator V4
* gpt2 training validation
* V4 training PEF-style fit/fail simulation
* safety-margin analysis for safe workload placement
* fp32 vs fp16 precision-memory analysis
* analytical int8/int4 quantization-memory estimation
* quantization metadata overhead simulation
* quantization-aware total-memory approximation

---

## Current Status

The project currently has three completed core phases:

1. **Inference memory prediction**
2. **Training memory prediction**
3. **Precision and quantization memory adaptation**

The inference phase studies memory behavior during text generation.

The training phase studies memory behavior during forward pass, loss computation, backward pass, optimizer step, and zero_grad.

The precision/quantization phase studies how dtype and analytical quantization assumptions affect parameter memory and estimated total memory.

The current implementation focuses on GPT-style models and has been validated on:

* `sshleifer/tiny-gpt2`
* `distilgpt2`
* `gpt2`

Upcoming extensions include:

* sparsity memory estimation
* model-parallel memory partitioning simulation
* CNN vs Transformer architecture comparison
* final combined project report

---

## Models Used

| Model                 | Purpose                                |
| --------------------- | -------------------------------------- |
| `sshleifer/tiny-gpt2` | pipeline validation                    |
| `distilgpt2`          | main memory-scaling experiments        |
| `gpt2`                | validation on a larger GPT-style model |

`sshleifer/tiny-gpt2` was mainly used to validate the logger and pipeline.

`distilgpt2` was used for meaningful inference/training memory-scaling experiments.

`gpt2` was used to test whether the estimators generalize beyond `distilgpt2`.

---

## What Was Profiled

The profiling pipeline records:

* model name
* GPU name
* batch size
* input tokens
* generated tokens
* dtype
* cache setting
* optimizer name
* peak allocated memory
* peak reserved memory
* final allocated memory
* final reserved memory
* runtime
* OOM status
* error message

For training workloads, memory is also logged at key training stages:

* before model load
* after model load
* after batch creation
* after optimizer creation
* after forward pass
* after loss computation
* after backward pass
* after optimizer step
* after zero_grad

Stage-level logging is important because peak training memory can occur temporarily during backward pass or optimizer step.

---

## Estimator Modules

The project currently includes:

* `BaseMemoryEstimator`
* `ImprovedBaseMemoryEstimator`
* `KVCacheEstimator`
* `PrecisionAwareEstimator`
* `AllocatorCorrectionEstimator`
* `CombinedInferenceEstimator`
* `OptimizerStateEstimator`
* `TrainingMemoryEstimator`
* `TrainingMemoryEstimatorV4`
* `QuantizationMemoryEstimator`

---

## Key Inference Results

### Precision

For `distilgpt2`, fp16 reduced peak allocated memory from **343.41 MB** to **181.25 MB** compared with fp32.

This is a **47.22% reduction**.

For `gpt2`, fp16 reduced peak allocated memory from **504.11 MB** to **264.78 MB**.

This is a **47.48% reduction**.

Average measured fp16 allocated-memory reduction across the tested models:

| Metric                             |  Value |
| ---------------------------------- | -----: |
| Average allocated-memory reduction | 47.35% |
| Minimum allocated-memory reduction | 47.22% |
| Maximum allocated-memory reduction | 47.48% |

The reduction is close to half, but not exactly 50%, because total runtime memory includes more than parameter tensors.

---

### Batch Size

For `distilgpt2`, increasing batch size from 1 to 8 increased total peak allocated memory from **335.36 MB** to **384.00 MB**.

Memory per sample decreased because model weights are shared across the batch.

---

### Inference Estimator

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

* **2.36% mean error** for peak allocated memory
* **2.36% mean error** for peak reserved memory

For `gpt2`, the estimator achieved:

* **2.50% mean error** for peak allocated memory
* **2.28% mean error** for peak reserved memory
* **2.66% allocated MRE**
* **2.06% reserved MRE**

This supports that the inference estimator generalizes beyond `distilgpt2`.

---

### Inference PEF-style Fit/Fail Simulation

Across 720 simulated fit/fail cases:

* overall accuracy: **94.17%**
* dangerous failure rate: **5.56%**
* conservative failure rate: **0.28%**

For `distilgpt2`:

* accuracy: **99.52%**
* dangerous failure rate: **0.00%**

---

## Key Training Results

### Training vs Inference

For `distilgpt2`, training used around **4.7x to 4.8x** more peak allocated memory than inference under comparable settings.

This shows that training memory requires separate modeling from inference memory.

Training memory includes additional components:

* gradients
* optimizer states
* backward temporary tensors
* optimizer-step memory behavior

---

### Sequence Length

For `distilgpt2` training with AdamW:

| Input tokens | Peak allocated memory | Peak reserved memory |
| -----------: | --------------------: | -------------------: |
|           32 |            1592.43 MB |            1700.0 MB |
|           64 |            1600.69 MB |            1738.0 MB |
|          128 |            1616.71 MB |            1726.0 MB |

Training memory increased with sequence length, but fixed components such as parameters, gradients, and optimizer states dominated total memory.

---

### Batch Size

For `distilgpt2` training with AdamW and input length 64:

| Batch size | Peak allocated memory | Peak reserved memory |
| ---------: | --------------------: | -------------------: |
|          1 |            1600.19 MB |            1738.0 MB |
|          2 |            1616.21 MB |            1726.0 MB |
|          4 |            1648.43 MB |            1768.0 MB |

Batch-size scaling was sublinear because parameters, gradients, and optimizer states are shared across the batch.

---

### Optimizer Comparison

For `distilgpt2` training with batch size 1 and input length 64:

| Optimizer | Peak allocated memory | Peak reserved memory | Final allocated memory |
| --------- | --------------------: | -------------------: | ---------------------: |
| SGD       |             956.49 MB |            1030.0 MB |              345.63 MB |
| AdamW     |            1600.94 MB |            1738.0 MB |              973.48 MB |

AdamW used **644.45 MB** more peak allocated memory than SGD.

The stage-wise logs showed that both optimizers had similar memory after backward, but AdamW jumped sharply after optimizer step.

For `gpt2`, the same pattern was observed. At 64 tokens, AdamW current allocated memory increased from **983.20 MB** after backward to **1934.12 MB** after optimizer step, a jump of about **950.92 MB**.

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

* parameter memory: **312.47 MB**
* estimated AdamW optimizer-state memory: **624.94 MB**
* observed AdamW-SGD peak allocated difference: **644.45 MB**
* relative error: **3.03%**

For `gpt2` in fp32:

* parameter memory: **474.70 MB**
* estimated AdamW optimizer-state memory: **949.40 MB**
* observed AdamW optimizer-step jump at 64 tokens: **950.92 MB**

This supports modeling AdamW optimizer state as approximately **2 × parameter memory**.

---

## TrainingMemoryEstimator Evolution

### V1

The first training estimator used:

* parameter memory
* gradient memory
* optimizer-state memory
* activation memory
* framework overhead

It underpredicted memory because it did not include backward temporary memory.

---

### V2

V2 added backward temporary memory correction:

```text
backward_temp_memory = parameter_memory × 0.65
```

For `distilgpt2`, V2 achieved:

* **2.98% allocated MRE**
* **3.50% reserved MRE**
* **3.46% allocated mean error**
* **3.87% reserved mean error**

---

### V3

V3 made the backward temporary correction optimizer-specific:

| Case              | Backward temporary factor |
| ----------------- | ------------------------: |
| AdamW             |                      0.65 |
| Adam              |                      0.65 |
| SGD               |                      0.35 |
| SGD with momentum |                      0.45 |

V3 improved `gpt2` SGD prediction but damaged `distilgpt2` SGD prediction.

This showed that optimizer-specific correction alone was not enough.

---

### V4

V4 adds optimizer-specific and model-size-aware correction:

| Case                            | Backward temporary factor |
| ------------------------------- | ------------------------: |
| AdamW                           |                      0.65 |
| Adam                            |                      0.65 |
| SGD below 100M parameters       |                      0.65 |
| SGD above/equal 100M parameters |                      0.35 |
| SGD with momentum               |                      0.45 |
| default                         |                      0.50 |

This fixed the `distilgpt2` SGD failure from V3 while preserving the `gpt2` SGD improvement.

---

## TrainingMemoryEstimator V4 Results

On the combined `distilgpt2` + `gpt2` training validation set, V4 achieved:

| Metric               | Value |
| -------------------- | ----: |
| allocated MRE        | 2.98% |
| allocated mean error | 3.66% |
| allocated max error  | 8.77% |
| reserved MRE         | 3.69% |
| reserved mean error  | 4.29% |
| reserved max error   | 9.52% |

V4 is the current candidate final training estimator.

---

### V2 vs V3 vs V4

| Estimator | Allocated MRE | Allocated mean error | Allocated max error | Reserved MRE | Reserved mean error | Reserved max error |
| --------- | ------------: | -------------------: | ------------------: | -----------: | ------------------: | -----------------: |
| V2        |         2.99% |                4.95% |              17.69% |        3.69% |               5.00% |             16.83% |
| V3        |         2.99% |                4.36% |              12.77% |        4.20% |               4.99% |             12.52% |
| V4        |         2.98% |                3.66% |               8.77% |        3.69% |               4.29% |              9.52% |

V4 reduced worst-case error while preserving good performance across both `distilgpt2` and `gpt2`.

---

## V4 Training PEF-style Fit/Fail Simulation

A PEF-style simulation was performed using V4 predicted reserved memory.

The goal was to test whether V4 can correctly decide if a training workload fits under a GPU memory limit.

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

This improves over the earlier training PEF result of **91.07% accuracy**.

---

### PEF by Model

| Model      | Accuracy | Dangerous failure rate | Conservative failure rate |
| ---------- | -------: | ---------------------: | ------------------------: |
| distilgpt2 |   96.05% |                  2.63% |                     1.32% |
| gpt2       |   96.49% |                  3.51% |                     0.00% |

---

### PEF by Optimizer

| Optimizer | Accuracy | Dangerous failure rate | Conservative failure rate |
| --------- | -------: | ---------------------: | ------------------------: |
| AdamW     |   96.32% |                  2.63% |                     1.05% |
| SGD       |   96.05% |                  3.95% |                     0.00% |

Most failures occurred near tight memory boundaries, especially around **1700 MB** and **2600 MB**.

At relaxed memory limits such as **2048 MB**, **3072 MB**, **4096 MB**, and **8192 MB**, predictions were fully correct.

---

## V4 Safety-margin Analysis

V4 without safety margin achieved high accuracy but still had dangerous underpredictions.

Safety margins were tested:

| Safety margin | Accuracy | Dangerous failures | Conservative failures |
| ------------: | -------: | -----------------: | --------------------: |
|            0% |   96.24% |                  8 |                     2 |
|            2% |   97.37% |                  4 |                     3 |
|            5% |   96.62% |                  1 |                     8 |
|          7.5% |   95.11% |                  0 |                    13 |
|           10% |   93.98% |                  0 |                    16 |
|           15% |   89.85% |                  0 |                    27 |

The highest raw accuracy was at **2%**, but it still had dangerous failures.

The safest recommended setting is:

```text
TrainingMemoryEstimator V4 + 7.5% safety margin
```

At 7.5% margin:

* accuracy: **95.11%**
* dangerous failures: **0**
* conservative failures: **13**

This shows the tradeoff between safety and utilization.

---

## Final Estimator Recommendations

| Use case                           | Recommended estimator                             | Reason                                                          |
| ---------------------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| LLM inference memory prediction    | `CombinedInferenceEstimator`                      | Strong low-error prediction on `distilgpt2` and `gpt2`          |
| LLM training raw memory prediction | `TrainingMemoryEstimator V4`                      | Best combined training estimator across `distilgpt2` and `gpt2` |
| Safe training workload placement   | `TrainingMemoryEstimator V4 + 7.5% safety margin` | Removes all dangerous underpredictions in current validation    |

---

## Precision and Quantization Adaptation

The precision and quantization phase extends the project toward optimization-aware memory estimation.

This phase includes:

* measured fp32 vs fp16 precision comparison
* analytical fp32/fp16/int8/int4 parameter-memory estimation
* metadata overhead simulation
* reusable `QuantizationMemoryEstimator`
* measured vs analytical comparison
* quantization-aware total-memory approximation

Important clarification:

```text
int8 and int4 results are analytical memory estimates, not real measured quantized model execution results.
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
| Average reduction | 47.35% |
| Minimum reduction | 47.22% |
| Maximum reduction | 47.48% |

The theoretical fp16 parameter-memory reduction is 50%, but measured runtime memory reduction is lower because runtime memory also includes activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

---

## Analytical Quantization Parameter-memory Estimates

The quantization estimator uses:

```text
parameter_memory_MB = num_parameters × bytes_per_parameter / 1024²
```

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

Raw quantized parameter memory is not the full story.

Grouped quantization may require metadata such as:

* scale values
* zero-points
* group-level metadata

The simulation assumes:

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

At group size 128:

| Dtype | Metadata overhead as % of raw quantized memory |
| ----- | ---------------------------------------------: |
| int8  |                                         3.125% |
| int4  |                                          6.25% |

---

## QuantizationMemoryEstimator

The project now includes a reusable analytical module:

```text
src/estimators/quantization_memory_estimator.py
```

It estimates:

* raw parameter memory
* metadata memory
* effective parameter memory
* reduction vs fp32
* metadata overhead percentage

Supported dtypes:

* fp32
* fp16
* int8
* int4

Supported metadata cases:

* none
* scale_only_fp16
* scale_plus_zero_point

This module is analytical and should not be described as real int8/int4 runtime profiling.

---

## Measured vs Analytical Precision Comparison

The project compares measured fp32/fp16 runtime memory against analytical fp32/fp16 parameter memory.

| Model      | Measured fp16 reduction | Theoretical fp16 parameter reduction |                    Gap |
| ---------- | ----------------------: | -----------------------------------: | ---------------------: |
| distilgpt2 |                  47.22% |                               50.00% | 2.78 percentage points |
| gpt2       |                  47.48% |                               50.00% | 2.52 percentage points |

Average gap:

```text
2.65 percentage points
```

This shows that parameter-memory reduction does not directly equal total runtime-memory reduction.

---

## Quantization-aware Total-memory Approximation

The project estimates total memory under quantized dtypes using:

```text
estimated_total_memory(dtype)
=
measured_fp32_total_memory
- fp32_parameter_memory
+ effective_parameter_memory(dtype)
```

Equivalently:

```text
estimated_total_memory(dtype)
=
non_parameter_memory
+ effective_parameter_memory(dtype)
```

The approximation assumes non-parameter memory remains unchanged from fp32.

---

### fp32 Runtime Memory Decomposition

| Model      | fp32 measured total | fp32 parameter memory | non-parameter memory | parameter fraction | non-parameter fraction |
| ---------- | ------------------: | --------------------: | -------------------: | -----------------: | ---------------------: |
| distilgpt2 |           343.41 MB |             312.47 MB |             30.94 MB |             90.99% |                  9.01% |
| gpt2       |           504.11 MB |             474.70 MB |             29.41 MB |             94.17% |                  5.83% |

The tested GPT-style inference workloads are parameter-dominated.

---

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

The fp16 approximation is close to measured fp16 behavior, especially for `gpt2`, supporting the decomposition method as a reasonable analytical approximation.

However, int8 and int4 remain analytical estimates only.

---

## Project Structure

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

plots/
    actual_vs_predicted_allocated.png
    actual_vs_predicted_reserved.png
    pef_failure_rate_by_limit.png
    pef_accuracy_by_model.png
    dangerous_failure_by_model.png

    v4_training_pef_accuracy_by_memory_limit.png
    v4_training_pef_failure_rate_by_memory_limit.png
    v4_training_pef_accuracy_by_model.png
    v4_training_pef_accuracy_by_optimizer.png

    precision_peak_allocated_comparison.png
    precision_peak_reserved_comparison.png
    precision_memory_reduction_percent.png

    quantization_parameter_memory_by_dtype.png
    quantization_memory_reduction_percent.png
    quantization_distilgpt2_gpt2_comparison.png
    quantization_effective_parameter_memory.png
    quantization_metadata_overhead_by_group_size.png
    quantization_effective_reduction_percent.png
    quantization_dtype_effective_memory.png
    quantization_measured_vs_parameter_memory.png
    quantization_dtype_reduction_summary.png
    quantization_total_memory_by_dtype.png
    quantization_parameter_vs_non_parameter.png
    quantization_total_memory_reduction.png

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
```

---

## Main Interpretation

The inference estimator generalizes well from `distilgpt2` to `gpt2`.

Training memory requires a separate estimator because training introduces gradients, optimizer states, backward temporary tensors, and optimizer-step memory behavior.

AdamW uses significantly more memory than SGD because it stores optimizer-state tensors. The observed AdamW memory jump closely matches the expected 2 × parameter-memory rule.

TrainingMemoryEstimator V4 is the strongest current training estimator because it combines optimizer-specific and model-size-aware backward temporary memory correction.

V4 also improves deployment-style fit/fail prediction, achieving **96.24% PEF-style accuracy** across 266 training placement cases.

For safe placement, V4 should use a **7.5% safety margin**, which removes dangerous underpredictions in the current validation set.

The precision/quantization phase shows that dtype and quantization are important memory-optimization levers, but parameter-memory savings do not directly equal total runtime-memory savings.

---

## Limitations

Current limitations:

* Experiments are limited to single-GPU Colab/T4-style runs.
* The main validation models are `distilgpt2` and `gpt2`.
* The 100M parameter threshold in V4 is empirical.
* Larger LLMs still need validation.
* Real model parallelism is not implemented yet.
* Sparsity modules are planned but not complete yet.
* CNN and Vision Transformer architecture comparison is planned but not complete yet.
* Reserved memory can vary across CUDA/PyTorch runtime environments.
* int8/int4 quantization results are analytical estimates only.
* No real int8/int4 quantized model execution has been profiled yet.
* Quantization latency and quality impact are not measured.
* Activation quantization and KV-cache quantization are not modeled yet.
* Quantization kernel packing/alignment overhead is not measured.

---

## Next Work

Planned next steps:

1. Summarize the precision and quantization adaptation phase.
2. Add sparsity memory estimation.
3. Add model-parallel memory partitioning simulation.
4. Add CNN vs Transformer memory comparison.
5. Prepare final combined project report.
6. Update final README after sparsity, model parallelism, and architecture comparison are complete.
7. Prepare final mentor/interview package.

---

## Final Current Status

The project currently has strong completed work in:

* LLM inference memory profiling
* LLM inference memory prediction
* gpt2 inference validation
* LLM training memory profiling
* optimizer-state estimation
* training memory prediction
* gpt2 training validation
* TrainingMemoryEstimator V4
* V4 training PEF-style fit/fail simulation
* V4 safety-margin analysis
* final estimator comparison
* fp32/fp16 precision-memory analysis
* analytical quantization-memory estimation
* quantization metadata overhead simulation
* reusable `QuantizationMemoryEstimator`
* quantization-aware total-memory approximation

The next phase extends the project further toward the official scope by adding sparsity, model-parallel memory estimation, and architecture comparison.
