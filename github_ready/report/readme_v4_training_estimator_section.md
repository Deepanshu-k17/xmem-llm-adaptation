## TrainingMemoryEstimator V4

`TrainingMemoryEstimatorV4` is the current candidate final training-memory estimator.

It estimates GPT-style training memory using:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- backward temporary memory
- framework overhead
- CUDA allocator padding

V4 adds optimizer-specific and model-size-aware backward temporary correction.

| Case | backward_temp_factor |
|---|---:|
| AdamW | 0.65 |
| Adam | 0.65 |
| SGD below 100M parameters | 0.65 |
| SGD above/equal 100M parameters | 0.35 |
| SGD with momentum | 0.45 |
| default | 0.50 |

On the combined distilgpt2 + gpt2 training validation set, V4 achieved:

- allocated MRE: 2.98%
- allocated mean error: 3.66%
- reserved MRE: 3.69%
- reserved mean error: 4.29%

V4 fixed the distilgpt2 SGD failure introduced by V3 while preserving the gpt2 SGD improvement.