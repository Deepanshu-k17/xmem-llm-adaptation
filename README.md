# xMem-inspired LLM Memory Profiling and Prediction

This project builds an xMem-inspired GPU memory profiling and prediction pipeline for Transformer/LLM inference workloads.

The goal is to study how peak GPU memory changes with model size, input length, generated length, batch size, cache setting, dtype, and allocator behavior, then build estimator modules that predict peak allocated and peak reserved memory.

## Current Status

The current implementation focuses on inference memory prediction.

Training memory profiling and optimizer-state estimation will be added next.

## Models Used

- `sshleifer/tiny-gpt2`
- `distilgpt2`

`sshleifer/tiny-gpt2` was mainly used for pipeline validation.

`distilgpt2` was used for more meaningful memory-scaling experiments.

## What Was Profiled

The profiling pipeline records:

- model name
- GPU name
- batch size
- input tokens
- generated tokens
- dtype
- cache setting
- peak allocated memory
- peak reserved memory
- final allocated memory
- final reserved memory
- runtime
- OOM status
- error message

## Estimator Modules

The project currently includes:

- `BaseMemoryEstimator`
- `ImprovedBaseMemoryEstimator`
- `KVCacheEstimator`
- `PrecisionAwareEstimator`
- `AllocatorCorrectionEstimator`
- `CombinedInferenceEstimator`

## Key Results

### Precision

For `distilgpt2`, fp16 reduced peak allocated memory from 343.41 MB to 181.25 MB.

This is a 47.22% reduction.

### Batch Size

For `distilgpt2`, increasing batch size from 1 to 8 increased total peak allocated memory from 335.36 MB to 384.00 MB.

Memory per sample decreased sharply because model weights are shared across the batch.

### Inference Estimator

For `distilgpt2`, the `CombinedInferenceEstimator` achieved:

- 2.36% mean error for peak allocated memory
- 2.36% mean error for peak reserved memory

### PEF-style Fit/Fail Simulation

Across 720 simulated fit/fail cases:

- overall accuracy: 94.17%
- dangerous failure rate: 5.56%
- conservative failure rate: 0.28%

For `distilgpt2`:

- accuracy: 99.52%
- dangerous failure rate: 0.00%

## Project Structure

```text
src/
    experiment_logger.py
    estimators/
        model_config_utils.py
        base_estimator.py
        improved_base_estimator.py
        kv_cache_estimator.py
        precision_estimator.py
        allocator_correction.py
        combined_inference_estimator.py

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

plots/
    all_rows_mre_comparison.png
    distilgpt2_mre_comparison.png
    all_vs_distilgpt2_mean_error.png
    actual_vs_predicted_allocated.png
    actual_vs_predicted_reserved.png
    pef_failure_rate_by_limit.png
    pef_accuracy_by_model.png
    dangerous_failure_by_model.png

report/
    inference_phase_report.md
    pef_style_simulation.md
```

## Main Interpretation

The estimator works well for `distilgpt2`, which is the more meaningful model in the current experiments.

`sshleifer/tiny-gpt2` is useful for validating the pipeline, but it is too small for realistic memory-estimation evaluation because framework and allocator overhead dominate its memory usage.

## Next Work

- training memory logger
- training sequence-length and batch-size experiments
- optimizer comparison: SGD vs AdamW
- optimizer-state memory estimator
- training-side MRE evaluation