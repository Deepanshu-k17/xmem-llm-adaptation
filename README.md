# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference and training workloads.

The goal is to study how peak GPU memory changes with model size, input length, generated length, batch size, cache setting, dtype, optimizer choice, and allocator behavior, then build estimator modules that predict peak allocated and peak reserved GPU memory.

The project currently includes:

* inference memory profiling
* inference memory estimators
* PEF-style fit/fail simulation for inference
* training memory profiling
* optimizer-state memory estimation
* training memory estimator
* training PEF-style fit/fail simulation
* safety-margin analysis for safer workload placement

---

## Current Status

The project has two main phases:

1. **Inference memory prediction**
2. **Training memory prediction**

The inference phase studies memory behavior during text generation.

The training phase studies memory behavior during forward pass, loss computation, backward pass, optimizer step, and zero_grad.

The current implementation includes both phases for GPT-style models.

---

## Models Used

* `sshleifer/tiny-gpt2`
* `distilgpt2`

`sshleifer/tiny-gpt2` was mainly used for pipeline validation.

`distilgpt2` was used for more meaningful memory-scaling experiments.

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

For training workloads, memory is also logged at different stages:

* before model load
* after model load
* after batch creation
* after optimizer creation
* after forward pass
* after loss computation
* after backward pass
* after optimizer step
* after zero_grad

This stage-level logging is important because peak training memory can happen temporarily during backward pass or optimizer step.

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

---

## Key Inference Results

### Precision

For `distilgpt2`, fp16 reduced peak allocated memory from **343.41 MB** to **181.25 MB** compared to fp32.

This is a **47.22% reduction**.

### Batch Size

For `distilgpt2`, increasing batch size from 1 to 8 increased total peak allocated memory from **335.36 MB** to **384.00 MB**.

Memory per sample decreased sharply because model weights are shared across the batch.

### Inference Estimator

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

* **2.36% mean error** for peak allocated memory
* **2.36% mean error** for peak reserved memory

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

This shows that training memory needs separate modeling from inference memory.

### Sequence Length

For `distilgpt2` training with AdamW:

| input tokens | peak allocated memory | peak reserved memory |
| -----------: | --------------------: | -------------------: |
|           32 |            1592.43 MB |            1700.0 MB |
|           64 |            1600.69 MB |            1738.0 MB |
|          128 |            1616.71 MB |            1726.0 MB |

Training memory increased with sequence length, but fixed components such as parameters, gradients, and optimizer states dominated the total memory.

### Batch Size

For `distilgpt2` training with AdamW and input length 64:

| batch size | peak allocated memory | peak reserved memory |
| ---------: | --------------------: | -------------------: |
|          1 |            1600.19 MB |            1738.0 MB |
|          2 |            1616.21 MB |            1726.0 MB |
|          4 |            1648.43 MB |            1768.0 MB |

Batch-size scaling was sublinear because parameters, gradients, and optimizer states are shared across the batch.

### Optimizer Comparison

For `distilgpt2` training with batch size 1 and input length 64:

| optimizer | peak allocated memory | peak reserved memory | final allocated memory |
| --------- | --------------------: | -------------------: | ---------------------: |
| SGD       |             956.49 MB |            1030.0 MB |              345.63 MB |
| AdamW     |            1600.94 MB |            1738.0 MB |              973.48 MB |

AdamW used **644.45 MB** more peak allocated memory than SGD.

The stage-wise logs showed that both optimizers had the same peak after backward, but AdamW jumped sharply after optimizer step. This shows that AdamW optimizer states are a major contributor to training memory.

---

## OptimizerStateEstimator

The `OptimizerStateEstimator` models optimizer-state memory using optimizer-specific factors:

| optimizer         | optimizer-state factor |
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

This supports modeling AdamW optimizer state as approximately **2 × parameter memory**.

---

## TrainingMemoryEstimator

The training estimator uses:

* parameter memory
* gradient memory
* optimizer-state memory
* activation memory
* backward temporary memory
* allocator padding

The first version underpredicted memory because it did not include backward temporary memory.

After adding backward temporary memory correction, the estimator improved significantly.

| estimator                  | allocated MRE | reserved MRE |
| -------------------------- | ------------: | -----------: |
| TrainingMemoryEstimator V1 |        15.66% |       16.12% |
| TrainingMemoryEstimator V2 |         2.98% |        3.50% |

`TrainingMemoryEstimator V2` achieved:

* **2.98% allocated MRE**
* **3.50% reserved MRE**
* **3.46% allocated mean error**
* **3.87% reserved mean error**

---

## Training PEF-style Fit/Fail Simulation

A PEF-style simulation was performed using predicted vs actual reserved memory.

Without safety margin:

* total cases: **112**
* accuracy: **91.07%**
* dangerous failures: **7**
* conservative failures: **3**

Most dangerous failures occurred near tight memory boundaries.

### Safety Margin

A safety margin was added to predicted reserved memory:

```text
safe_predicted_reserved_MB = predicted_peak_reserved_MB × (1 + safety_margin)
```

| safety margin | accuracy | dangerous failures | conservative failures |
| ------------: | -------: | -----------------: | --------------------: |
|            0% |   91.07% |                  7 |                     3 |
|            5% |   91.96% |                  0 |                     9 |
|           10% |   86.61% |                  0 |                    15 |
|           15% |   85.71% |                  0 |                    16 |

A **5% safety margin** gave the best tradeoff.

It removed all dangerous failures while slightly improving overall accuracy.

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

results/
    inference_clean.csv
    inference_summary.csv
    estimator_comparison.csv
    reserved_memory_summary.csv
    key_findings.csv
    pef_summary.csv
    pef_by_model.csv
    pef_by_memory_limit.csv
    realistic_gpu_limit_summary.csv

    training_phase_key_metrics.csv
    training_phase_key_findings.csv
    training_estimator_v2_predictions.csv
    training_estimator_v2_metrics.csv
    training_estimator_v1_vs_v2_metrics.csv
    training_pef_summary.csv
    training_pef_safety_margin_summary.csv
    training_pef_best_safety_margin.csv

plots/
    actual_vs_predicted_allocated.png
    actual_vs_predicted_reserved.png
    pef_failure_rate_by_limit.png
    pef_accuracy_by_model.png
    dangerous_failure_by_model.png

    training_estimator_v1_vs_v2_allocated_mre.png
    training_estimator_v1_vs_v2_reserved_mre.png
    training_actual_vs_predicted_allocated.png
    training_pef_accuracy_vs_safety_margin.png
    training_pef_dangerous_failure_vs_safety_margin.png
    training_pef_tradeoff_vs_safety_margin.png

report/
    inference_phase_report.md
    pef_style_simulation.md
    training_phase_report.md
    optimizer_state_estimator_report.md
    training_memory_estimator_report.md
    training_estimator_improvement_report.md
    training_pef_simulation_report.md
    training_pef_safety_margin_report.md
```

---

## Main Interpretation

The inference estimator works well for `distilgpt2`, which is the more meaningful model in the current experiments.

`sshleifer/tiny-gpt2` is useful for validating the pipeline, but it is too small for realistic memory-estimation evaluation because framework and allocator overhead dominate its memory usage.

For training, optimizer choice is critical. AdamW uses significantly more memory than SGD because of optimizer-state tensors.

The training estimator becomes much more accurate after adding backward temporary memory correction.

The 5% safety margin makes fit/fail prediction safer by removing dangerous underprediction in the current training dataset.

---

## Limitations

Current limitations:

* Experiments are limited to single-GPU Colab/T4-style runs.
* The main meaningful model is `distilgpt2`.
* Larger GPT-style models still need validation.
* The backward temporary memory correction is empirical.
* The safety margin is evaluated on a small dataset.
* Real model parallelism is not implemented.
* Sparsity and quantization training experiments are not yet included.

---

## Next Work

Planned next steps:

* validate inference estimator on `gpt2`
* validate training estimator on `gpt2`
* test whether optimizer-state and backward-temp formulas generalize
* add more model-generalization analysis
* improve README and runnable examples
* prepare a final combined report after additional validation
