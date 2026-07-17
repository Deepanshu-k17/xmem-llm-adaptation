## Training Memory Profiling and Estimation

This project includes a training-memory phase for GPT-style models.

The training logger records CUDA memory at major training stages:

- model load
- batch creation
- optimizer creation
- forward pass
- loss computation
- backward pass
- optimizer step
- zero_grad

### Key Training Findings

For `distilgpt2`, training used around 4.7x to 4.8x more peak memory than inference under comparable token settings.

Batch-size scaling was sublinear. Peak allocated memory increased from 1600.19 MB at batch size 1 to 1648.43 MB at batch size 4, because model parameters, gradients, and optimizer states are shared across the batch.

Optimizer choice had a major impact. AdamW used 1600.94 MB peak allocated memory, while SGD used 956.49 MB. The difference was mainly caused by AdamW optimizer states.

### Training Estimator

The training estimator uses:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- backward temporary memory
- allocator padding

TrainingMemoryEstimator V2 achieved:

- 2.98% allocated MRE
- 3.50% reserved MRE

### Fit/Fail Simulation

A PEF-style fit/fail simulation was performed using predicted vs actual reserved memory.

Without safety margin:

- accuracy: 91.07%
- dangerous failures: 7

With a 5% safety margin:

- accuracy: 91.96%
- dangerous failures: 0

This shows that a small safety margin makes the estimator safer for scheduling-style decisions.