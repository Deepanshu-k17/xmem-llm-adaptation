# Inference Phase Report

## Goal

The goal of the inference phase was to build an xMem-inspired memory profiling and prediction pipeline for LLM inference workloads.

The project measured and estimated GPU memory across:

- model size
- input token length
- generated token length
- batch size
- cache setting
- dtype / precision
- allocator behavior

## Models Used

- sshleifer/tiny-gpt2
- distilgpt2

tiny-gpt2 was mainly used to validate the profiling pipeline.

distilgpt2 was used for more meaningful memory-scaling analysis.

## Main Profiling Findings

### Model Size

At 512 input tokens:

- tiny-gpt2 peak allocated memory: 12.53 MB
- distilgpt2 peak allocated memory: 384.01 MB

distilgpt2 used about 30.6x more memory.

### Input Token Scaling

For distilgpt2, increasing input tokens from 16 to 512 increased peak allocated memory from 331.22 MB to 384.01 MB.

### Generated Token Scaling

For distilgpt2, increasing max_new_tokens from 8 to 256 increased peak allocated memory from 335.36 MB to 341.35 MB.

### Batch Size Scaling

For distilgpt2, increasing batch size from 1 to 8 increased peak allocated memory from 335.36 MB to 384.00 MB.

Memory per sample decreased sharply because model weights are shared across the batch.

### Precision

fp16 reduced distilgpt2 peak allocated memory from 343.41 MB to 181.25 MB.

This was a 47.22% reduction.

### Allocator Behavior

For distilgpt2, median allocator padding ratio was 0.0854.

For tiny-gpt2, median allocator padding ratio was 1.08, showing that tiny models are dominated by framework and allocator overhead.

## Estimator Modules Built

The following modules were built:

1. BaseMemoryEstimator
2. ImprovedBaseMemoryEstimator
3. KVCacheEstimator
4. PrecisionAwareEstimator
5. AllocatorCorrectionEstimator
6. CombinedInferenceEstimator

## Key Inference Results

The best all-row allocated-memory MRE was achieved by the Improved + KV estimator:

- all-row MRE: 3.75%

The CombinedInferenceEstimator is the most complete pipeline because it predicts both peak allocated and peak reserved memory.

For distilgpt2, the CombinedInferenceEstimator achieved:

- allocated mean error: 2.36%
- reserved mean error: 2.36%

## Main Interpretation

The estimator works well for distilgpt2, which is the more meaningful model in the current experiments.

tiny-gpt2 is useful for validating the pipeline, but it is too small for realistic memory-estimation evaluation because framework and allocator overhead dominate its memory usage.

## Conclusion

The inference phase successfully produced a modular LLM memory prediction pipeline.

The estimator works well for distilgpt2, while tiny-gpt2 should be treated mainly as a pipeline validation model rather than a realistic memory-prediction target.

The next phase is training-memory profiling and optimizer-state estimation.
