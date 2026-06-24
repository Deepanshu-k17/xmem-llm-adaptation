<<<<<<< Updated upstream
# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference and training workloads.

The goal is to study how peak GPU memory changes with model size, input length, generated length, batch size, cache setting, dtype, optimizer choice, training stage, and CUDA allocator behavior, then build estimator modules that predict peak allocated and peak reserved GPU memory.

The project currently includes:

* inference memory profiling
* inference memory estimators
* gpt2 inference validation
* PEF-style fit/fail simulation for inference
* training memory profiling
* optimizer-state memory estimation
* training memory estimator V4
* gpt2 training validation
* V4 training PEF-style fit/fail simulation
* safety-margin analysis for safer workload placement

---

## Current Status

The project currently has two completed core phases:

1. **Inference memory prediction**
2. **Training memory prediction**

The inference phase studies memory behavior during text generation.

The training phase studies memory behavior during forward pass, loss computation, backward pass, optimizer step, and zero_grad.

The current implementation focuses on GPT-style models and has been validated on both `distilgpt2` and `gpt2`.

Upcoming extensions include:

* precision and quantization memory estimation
* sparsity memory estimation
* model-parallel memory partitioning
* CNN vs Transformer architecture comparison

These extensions are planned to align the project more closely with xMem-style adaptation for large model workloads.

---

## Models Used

* `sshleifer/tiny-gpt2`
* `distilgpt2`
* `gpt2`

`sshleifer/tiny-gpt2` was mainly used for pipeline validation.

`distilgpt2` was used for meaningful memory-scaling experiments.

`gpt2` was used to validate whether the estimators generalize to a larger GPT-style model.

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
* `TrainingMemoryEstimatorV4`

---

## Key Inference Results

### Precision

For `distilgpt2`, fp16 reduced peak allocated memory from **343.41 MB** to **181.25 MB** compared to fp32.

This is a **47.22% reduction**.

For `gpt2`, fp16 reduced peak allocated memory from **504.11 MB** to **264.78 MB** for the same batch/input/output setting.

This is approximately a **47.48% reduction**.

### Batch Size

For `distilgpt2`, increasing batch size from 1 to 8 increased total peak allocated memory from **335.36 MB** to **384.00 MB**.

Memory per sample decreased because model weights are shared across the batch.

### Inference Estimator

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

* **2.36% mean error** for peak allocated memory
* **2.36% mean error** for peak reserved memory

For `gpt2`, the estimator achieved:

* **2.50% mean error** for peak allocated memory
* **2.28% mean error** for peak reserved memory
* **2.66% allocated MRE**
* **2.06% reserved MRE**

This supports that the inference estimator generalizes beyond `distilgpt2` to a larger GPT-style model.

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

Training memory includes additional components:

* gradients
* optimizer states
* backward temporary tensors
* optimizer-step memory behavior

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

The stage-wise logs showed that both optimizers had similar memory after backward, but AdamW jumped sharply after optimizer step. This shows that AdamW optimizer states are a major contributor to training memory.

For `gpt2`, the same behavior was observed. At 64 tokens, AdamW current allocated memory increased from **983.20 MB** after backward to **1934.12 MB** after optimizer step, a jump of about **950.92 MB**.

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

### V3

V3 made the backward temporary correction optimizer-specific:

| case              | backward_temp_factor |
| ----------------- | -------------------: |
| AdamW             |                 0.65 |
| Adam              |                 0.65 |
| SGD               |                 0.35 |
| SGD with momentum |                 0.45 |

V3 improved `gpt2` SGD prediction but damaged `distilgpt2` SGD prediction.

This showed that optimizer-specific correction alone was not enough.

### V4

V4 adds optimizer-specific and model-size-aware correction.

| case                            | backward_temp_factor |
| ------------------------------- | -------------------: |
| AdamW                           |                 0.65 |
| Adam                            |                 0.65 |
| SGD below 100M parameters       |                 0.65 |
| SGD above/equal 100M parameters |                 0.35 |
| SGD with momentum               |                 0.45 |
| default                         |                 0.50 |

This fixed the `distilgpt2` SGD failure from V3 while preserving the `gpt2` SGD improvement.

---

## TrainingMemoryEstimator V4 Results

On the combined `distilgpt2` + `gpt2` training validation set, V4 achieved:

| metric               | value |
| -------------------- | ----: |
| allocated MRE        | 2.98% |
| allocated mean error | 3.66% |
| allocated max error  | 8.77% |
| reserved MRE         | 3.69% |
| reserved mean error  | 4.29% |
| reserved max error   | 9.52% |

V4 is the current candidate final training estimator.

### V2 vs V3 vs V4

| estimator | allocated MRE | allocated mean error | allocated max error | reserved MRE | reserved mean error | reserved max error |
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

| metric                    |  value |
| ------------------------- | -----: |
| total cases               |    266 |
| correct cases             |    256 |
| dangerous failures        |      8 |
| conservative failures     |      2 |
| accuracy                  | 96.24% |
| dangerous failure rate    |  3.01% |
| conservative failure rate |  0.75% |

This improves over the earlier training PEF result of **91.07% accuracy**.

### PEF by Model

| model      | accuracy | dangerous failure rate | conservative failure rate |
| ---------- | -------: | ---------------------: | ------------------------: |
| distilgpt2 |   96.05% |                  2.63% |                     1.32% |
| gpt2       |   96.49% |                  3.51% |                     0.00% |

### PEF by Optimizer

| optimizer | accuracy | dangerous failure rate | conservative failure rate |
| --------- | -------: | ---------------------: | ------------------------: |
| AdamW     |   96.32% |                  2.63% |                     1.05% |
| SGD       |   96.05% |                  3.95% |                     0.00% |

Most failures occurred near tight memory boundaries, especially around **1700 MB** and **2600 MB**.

At relaxed memory limits such as **2048 MB**, **3072 MB**, **4096 MB**, and **8192 MB**, predictions were fully correct.

The remaining issue is that **8 dangerous underpredictions still exist**, so the next step is V4 safety-margin simulation.

---

## Earlier Training Safety-margin Analysis

A previous safety-margin simulation was performed before V4 using predicted reserved memory.

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

A **5% safety margin** gave the best tradeoff in the earlier training dataset.

The next step is to repeat this safety-margin simulation using V4 predictions.

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
```

---

## Main Interpretation

The inference estimator generalizes well from `distilgpt2` to `gpt2`.

Training memory requires a separate estimator because training introduces gradients, optimizer states, backward temporary tensors, and optimizer-step memory behavior.

AdamW uses significantly more memory than SGD because it stores optimizer-state tensors. The observed AdamW memory jump closely matches the expected 2 × parameter-memory rule.

TrainingMemoryEstimator V4 is the strongest current training estimator because it combines optimizer-specific and model-size-aware backward temporary memory correction.

V4 also improves deployment-style fit/fail prediction, achieving **96.24% PEF-style accuracy** across 266 training placement cases.

Reserved memory is important for fit/fail prediction because CUDA allocator behavior can cause a workload to fail even when allocated memory alone looks safe.

---

## Limitations

Current limitations:

* Experiments are limited to single-GPU Colab/T4-style runs.
* The main validation models are `distilgpt2` and `gpt2`.
* The 100M parameter threshold in V4 is empirical.
* Larger LLMs still need validation.
* V4 safety-margin simulation is still pending.
* Real model parallelism is not implemented yet.
* Sparsity and quantization modules are planned but not complete yet.
* CNN and Vision Transformer architecture comparison is planned but not complete yet.
* Reserved memory can vary across CUDA/PyTorch runtime environments.

---

## Next Work

Planned next steps:

1. Run V4 safety-margin analysis.
2. Compare inference PEF vs training PEF.
3. Add precision and quantization memory estimation.
4. Add sparsity memory estimation.
5. Add model-parallel memory partitioning simulation.
6. Add CNN vs Transformer memory comparison.
7. Prepare final combined project report after all technical modules are complete.
8. Update final README after quantization, sparsity, model parallelism, and architecture comparison are done.
9. Prepare final mentor/interview package.

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

The next phase extends the project toward the official scope by adding V4 safety margins, quantization, sparsity, model-parallel memory estimation, and CNN/Transformer architecture comparison.
=======
# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference and training workloads.

The goal is to study how peak GPU memory changes with model size, input length, generated length, batch size, cache setting, dtype, optimizer choice, training stage, and CUDA allocator behavior, then build estimator modules that predict peak allocated and peak reserved GPU memory.

The project currently includes:

* inference memory profiling
* inference memory estimators
* gpt2 inference validation
* PEF-style fit/fail simulation for inference
* training memory profiling
* optimizer-state memory estimation
* training memory estimator V4
* gpt2 training validation
* training PEF-style fit/fail simulation
* safety-margin analysis for safer workload placement

---

## Current Status

The project currently has two completed core phases:

1. **Inference memory prediction**
2. **Training memory prediction**

The inference phase studies memory behavior during text generation.

The training phase studies memory behavior during forward pass, loss computation, backward pass, optimizer step, and zero_grad.

The current implementation focuses on GPT-style models and has been validated on both `distilgpt2` and `gpt2`.

Upcoming extensions include:

* precision and quantization memory estimation
* sparsity memory estimation
* model-parallel memory partitioning
* CNN vs Transformer architecture comparison

These extensions are planned to align the project more closely with xMem-style adaptation for large model workloads.

---

## Models Used

* `sshleifer/tiny-gpt2`
* `distilgpt2`
* `gpt2`

`sshleifer/tiny-gpt2` was mainly used for pipeline validation.

`distilgpt2` was used for meaningful memory-scaling experiments.

`gpt2` was used to validate whether the estimators generalize to a larger GPT-style model.

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
* `TrainingMemoryEstimatorV4`

---

## Key Inference Results

### Precision

For `distilgpt2`, fp16 reduced peak allocated memory from **343.41 MB** to **181.25 MB** compared to fp32.

This is a **47.22% reduction**.

For `gpt2`, fp16 reduced peak allocated memory from **504.11 MB** to **264.78 MB** for the same batch/input/output setting.

This is approximately a **47.48% reduction**.

### Batch Size

For `distilgpt2`, increasing batch size from 1 to 8 increased total peak allocated memory from **335.36 MB** to **384.00 MB**.

Memory per sample decreased because model weights are shared across the batch.

### Inference Estimator

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

* **2.36% mean error** for peak allocated memory
* **2.36% mean error** for peak reserved memory

For `gpt2`, the estimator achieved:

* **2.50% mean error** for peak allocated memory
* **2.28% mean error** for peak reserved memory
* **2.66% allocated MRE**
* **2.06% reserved MRE**

This supports that the inference estimator generalizes beyond `distilgpt2` to a larger GPT-style model.

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

Training memory includes additional components:

* gradients
* optimizer states
* backward temporary tensors
* optimizer-step memory behavior

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

The stage-wise logs showed that both optimizers had similar memory after backward, but AdamW jumped sharply after optimizer step. This shows that AdamW optimizer states are a major contributor to training memory.

For `gpt2`, the same behavior was observed. At 64 tokens, AdamW current allocated memory increased from **983.20 MB** after backward to **1934.12 MB** after optimizer step, a jump of about **950.92 MB**.

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

### V3

V3 made the backward temporary correction optimizer-specific:

| case              | backward_temp_factor |
| ----------------- | -------------------: |
| AdamW             |                 0.65 |
| Adam              |                 0.65 |
| SGD               |                 0.35 |
| SGD with momentum |                 0.45 |

V3 improved `gpt2` SGD prediction but damaged `distilgpt2` SGD prediction.

This showed that optimizer-specific correction alone was not enough.

### V4

V4 adds optimizer-specific and model-size-aware correction.

| case                            | backward_temp_factor |
| ------------------------------- | -------------------: |
| AdamW                           |                 0.65 |
| Adam                            |                 0.65 |
| SGD below 100M parameters       |                 0.65 |
| SGD above/equal 100M parameters |                 0.35 |
| SGD with momentum               |                 0.45 |
| default                         |                 0.50 |

This fixed the `distilgpt2` SGD failure from V3 while preserving the `gpt2` SGD improvement.

---

## TrainingMemoryEstimator V4 Results

On the combined `distilgpt2` + `gpt2` training validation set, V4 achieved:

| metric               | value |
| -------------------- | ----: |
| allocated MRE        | 2.98% |
| allocated mean error | 3.66% |
| allocated max error  | 8.77% |
| reserved MRE         | 3.69% |
| reserved mean error  | 4.29% |
| reserved max error   | 9.52% |

V4 is the current candidate final training estimator.

### V2 vs V3 vs V4

| estimator | allocated MRE | allocated mean error | allocated max error | reserved MRE | reserved mean error | reserved max error |
| --------- | ------------: | -------------------: | ------------------: | -----------: | ------------------: | -----------------: |
| V2        |         2.99% |                4.95% |              17.69% |        3.69% |               5.00% |             16.83% |
| V3        |         2.99% |                4.36% |              12.77% |        4.20% |               4.99% |             12.52% |
| V4        |         2.98% |                3.66% |               8.77% |        3.69% |               4.29% |              9.52% |

V4 reduced worst-case error while preserving good performance across both `distilgpt2` and `gpt2`.

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

The next step is to repeat this PEF-style simulation using the final V4 training estimator.

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

    gpt2_inference_validation.csv
    gpt2_inference_estimator_predictions.csv
    gpt2_inference_estimator_metrics.csv

    training_estimator_v4_predictions.csv
    training_estimator_v4_metrics.csv
    training_estimator_v4_error_by_model.csv
    training_estimator_v4_error_by_optimizer.csv
    training_estimator_v4_error_by_sequence_length.csv
    training_estimator_v2_v3_v4_metrics.csv

report/
    inference_phase_report.md
    pef_style_simulation.md
    training_phase_report.md
    optimizer_state_estimator_report.md
    training_memory_estimator_report.md
    training_estimator_improvement_report.md
    training_estimator_v4_model_size_correction_report.md
    final_training_estimator_report.md
```

---

## Main Interpretation

The inference estimator generalizes well from `distilgpt2` to `gpt2`.

Training memory requires a separate estimator because training introduces gradients, optimizer states, backward temporary tensors, and optimizer-step memory behavior.

AdamW uses significantly more memory than SGD because it stores optimizer-state tensors. The observed AdamW memory jump closely matches the expected 2 × parameter-memory rule.

TrainingMemoryEstimator V4 is the strongest current training estimator because it combines optimizer-specific and model-size-aware backward temporary memory correction.

Reserved memory is important for fit/fail prediction because CUDA allocator behavior can cause a workload to fail even when allocated memory alone looks safe.

---

## Limitations

Current limitations:

* Experiments are limited to single-GPU Colab/T4-style runs.
* The main validation models are `distilgpt2` and `gpt2`.
* The 100M parameter threshold in V4 is empirical.
* Larger LLMs still need validation.
* Real model parallelism is not implemented yet.
* Sparsity and quantization modules are planned but not complete yet.
* CNN and Vision Transformer architecture comparison is planned but not complete yet.
* Reserved memory can vary across CUDA/PyTorch runtime environments.

---

## Next Work

Planned next steps:

1. Run V4 training PEF-style fit/fail simulation.
2. Run V4 safety-margin analysis.
3. Compare inference PEF vs training PEF.
4. Add precision and quantization memory estimation.
5. Add sparsity memory estimation.
6. Add model-parallel memory partitioning simulation.
7. Add CNN vs Transformer memory comparison.
8. Prepare final combined project report after all technical modules are complete.
9. Update final README after quantization, sparsity, model parallelism, and architecture comparison are done.
10. Prepare final mentor/interview package.

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
* PEF-style simulation and safety-margin analysis

The next phase extends the project toward the official scope by adding quantization, sparsity, model-parallel memory estimation, and CNN/Transformer architecture comparison.
>>>>>>> Stashed changes
