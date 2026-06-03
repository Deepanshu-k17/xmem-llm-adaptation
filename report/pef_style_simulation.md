# PEF-style Simulation Report

## Goal

The goal of this simulation was to test whether the CombinedInferenceEstimator can correctly predict whether a workload fits inside a given GPU memory limit.

This goes beyond numeric memory error and checks scheduling-style fit/fail behavior.

## Memory Used

The simulation used peak reserved memory.

Actual memory:

actual_peak_reserved_MB

Predicted memory:

predicted_peak_reserved_MB

Reserved memory was used because it better reflects PyTorch CUDA allocator behavior and real GPU memory pressure.

## Definitions

### Dangerous Failure

A dangerous failure happens when:

predicted_reserved_MB <= memory_limit_MB

but

actual_reserved_MB > memory_limit_MB

This means the estimator predicts that the workload will fit, but it actually exceeds the memory limit.

This can cause OOM.

### Conservative Failure

A conservative failure happens when:

predicted_reserved_MB > memory_limit_MB

but

actual_reserved_MB <= memory_limit_MB

This means the estimator rejects a workload that would actually fit.

This wastes capacity, but it is safer than dangerous underprediction.

## Memory Limits Tested

The simulation tested both artificial small limits and realistic GPU limits:

- 16 MB
- 24 MB
- 26 MB
- 32 MB
- 64 MB
- 128 MB
- 190 MB
- 256 MB
- 360 MB
- 400 MB
- 512 MB
- 1024 MB
- 2048 MB
- 4096 MB
- 8192 MB

## Overall Result

Across 720 simulated cases:

- correct predictions: 678
- dangerous failures: 40
- conservative failures: 2
- overall accuracy: 94.17%
- dangerous failure rate: 5.56%
- conservative failure rate: 0.28%

## Model-wise Result

For distilgpt2:

- accuracy: 99.52%
- dangerous failure rate: 0.00%
- conservative failure rate: 0.48%

For tiny-gpt2:

- accuracy: 86.67%
- dangerous failure rate: 13.33%
- conservative failure rate: 0.00%

## Main Interpretation

The estimator is reliable for distilgpt2 scheduling-style decisions.

For distilgpt2, there were no dangerous failures.

The only distilgpt2 failures were conservative failures near a 360 MB memory limit, where the estimator slightly overpredicted memory.

tiny-gpt2 caused dangerous failures because actual reserved memory was 26 MB while predicted reserved memory was around 12 MB.

This confirms that tiny-gpt2 needs a reserved-memory floor or special handling.

## Realistic GPU Limits

For 4 GB and 8 GB memory limits, all tested workloads were correctly predicted to fit.

This is expected because the current models are small compared to these GPU limits.

## Conclusion

The PEF-style simulation shows that the estimator is useful for fit/fail memory prediction.

The main result is that distilgpt2 achieved 99.52% fit/fail accuracy with 0% dangerous failures.

The next phase should include larger models or training workloads to stress realistic GPU limits more seriously.
