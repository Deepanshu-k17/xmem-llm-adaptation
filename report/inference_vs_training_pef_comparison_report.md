# Inference vs Training PEF Comparison Report

## 1. Goal

This report compares inference and training PEF-style fit/fail prediction results.

The goal is to understand whether memory prediction is more reliable for inference or training, and how safety margins affect deployment-style placement decisions.

## 2. Why This Comparison Matters

MRE measures numeric prediction error, but PEF-style evaluation checks whether the estimator makes the correct placement decision under a memory limit.

This matters for shared GPU clusters because an unsafe prediction can cause OOM.

## 3. Compared Settings

The comparison includes:

- inference PEF using the CombinedInferenceEstimator
- training PEF using TrainingMemoryEstimator V4 without safety margin
- training PEF using TrainingMemoryEstimator V4 with the recommended 7.5% safety margin

## 4. Main Comparison Table

| phase     | estimator                                 |   safety_margin_percent |   total_cases |   correct_cases |   dangerous_failures |   conservative_failures |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent |
|:----------|:------------------------------------------|------------------------:|--------------:|----------------:|---------------------:|------------------------:|-------------------:|---------------------------------:|------------------------------------:|
| inference | CombinedInferenceEstimator                |                     0   |           720 |             678 |                   40 |                       2 |            94.17   |                          5.56    |                             0.28    |
| training  | TrainingMemoryEstimatorV4                 |                     0   |           266 |             256 |                    8 |                       2 |            96.2406 |                          3.00752 |                             0.75188 |
| training  | TrainingMemoryEstimatorV4 + safety margin |                     7.5 |           266 |             253 |                    0 |                      13 |            95.1128 |                          0       |                             4.88722 |

## 5. Safety Comparison

| comparison              | phase     |   safety_margin_percent |   accuracy_percent |   dangerous_failure_rate_percent |   conservative_failure_rate_percent | interpretation                                                                      |
|:------------------------|:----------|------------------------:|-------------------:|---------------------------------:|------------------------------------:|:------------------------------------------------------------------------------------|
| inference_no_margin     | inference |                     0   |            94.17   |                          5.56    |                             0.28    | Baseline inference PEF performance.                                                 |
| training_v4_no_margin   | training  |                     0   |            96.2406 |                          3.00752 |                             0.75188 | V4 improves training PEF accuracy but still has dangerous failures.                 |
| training_v4_safe_margin | training  |                     7.5 |            95.1128 |                          0       |                             4.88722 | Recommended safe training placement setting because dangerous failures are removed. |

## 6. Key Findings

|   finding_id | finding                                                                 | evidence                                                                                      | interpretation                                                                            |
|-------------:|:------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|
|            1 | Training V4 PEF accuracy is higher than overall inference PEF accuracy. | Training V4 achieved 96.24% accuracy, while overall inference PEF accuracy was 94.17%.        | V4 improved training fit/fail prediction despite training being more complex.             |
|            2 | Training V4 still needs safety margin for deployment safety.            | Without margin, V4 had 8 dangerous failures. At 7.5% margin, dangerous failures dropped to 0. | Raw accuracy alone is not enough for safe GPU placement.                                  |
|            3 | The 7.5% training margin trades accuracy for safety.                    | Accuracy dropped from 96.24% to 95.11%, but dangerous failures dropped from 8 to 0.           | This is acceptable when avoiding OOM matters more than maximizing utilization.            |
|            4 | PEF is stricter than MRE.                                               | Small reserved-memory errors near memory limits caused dangerous fit/fail mistakes.           | A numerically accurate estimator can still be unsafe near scheduling boundaries.          |
|            5 | Reserved memory is the correct basis for placement simulation.          | PEF simulation used actual and predicted peak reserved memory rather than allocated memory.   | CUDA allocator reservation determines whether a workload may hit memory-limit boundaries. |

## 7. Interpretation

Training memory is more complex than inference memory because it includes gradients, optimizer states, backward temporary tensors, and optimizer-step behavior.

Despite this, TrainingMemoryEstimator V4 achieved strong PEF accuracy.

Without safety margin, V4 training PEF achieved 96.24% accuracy but still had 8 dangerous failures.

With a 7.5% safety margin, dangerous failures were reduced to zero, while accuracy remained 95.11%.

This shows that safety margin is necessary when memory prediction is used for real placement decisions.

## 8. Why Raw Accuracy Is Not Enough

The 2% safety margin achieved the highest raw accuracy in the Day 38 analysis, but still had dangerous failures.

For GPU placement, zero dangerous failures is more important than maximum accuracy.

Therefore, the recommended setting is V4 with 7.5% safety margin.

## 9. Conclusion

Inference and training PEF both show that estimator quality should be evaluated not only by MRE but also by fit/fail behavior.

Training V4 provides strong fit/fail accuracy, and using a 7.5% safety margin makes it safer for deployment-style workload placement.

The next stage of the project will extend beyond basic LLM inference/training memory prediction toward precision/quantization, sparsity, model-parallel partitioning, and architecture comparison.