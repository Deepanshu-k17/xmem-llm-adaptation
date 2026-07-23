# Final Training Estimator Report

## 1. Purpose

This report summarizes the final training-memory estimator module developed so far in the xMem-inspired LLM memory profiling project.

This is not the final project report. It only locks the current training estimator work before moving to PEF, quantization, sparsity, model-parallelism, and architecture comparison.

## 2. Why Training Memory Needed a Separate Estimator

Inference memory is mainly affected by model parameters, activations, generated tokens, KV cache, dtype, and allocator behavior.

Training memory has additional components:

- gradients
- optimizer states
- backward temporary tensors
- optimizer-step memory behavior

Because of these extra components, training memory required a separate estimator.

## 3. Training Logger

The training logger records CUDA memory at these stages:

- before_model_load
- after_model_load
- after_batch_creation
- after_optimizer_creation
- after_forward
- after_loss
- after_backward
- after_optimizer_step
- after_zero_grad

This stage-wise logging showed where training memory peaks occur.

## 4. Estimator Components

The training estimator models:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- backward temporary memory
- framework overhead
- CUDA allocator padding

## 5. Optimizer State Estimation

Optimizer-state memory was modeled using optimizer-specific factors:

| Optimizer | Optimizer-state memory |
|---|---:|
| SGD | 0 x parameter memory |
| SGD with momentum | 1 x parameter memory |
| Adam | 2 x parameter memory |
| AdamW | 2 x parameter memory |

For distilgpt2, the estimated AdamW optimizer-state memory was 624.94 MB, close to the observed AdamW-SGD peak allocated difference of 644.45 MB.

For gpt2, the AdamW current allocated memory jump after optimizer_step was about 950.92 MB, almost matching the estimated optimizer-state memory of 949.40 MB.

## 6. Estimator Evolution

### TrainingMemoryEstimator V1

V1 used parameter memory, gradient memory, optimizer-state memory, activation memory, and framework overhead.

It underpredicted training memory because it missed backward temporary memory.

### TrainingMemoryEstimator V2

V2 added backward temporary memory correction:

`backward_temp_memory = parameter_memory x 0.65`

V2 worked well on distilgpt2 but had higher error on gpt2 SGD.

### TrainingMemoryEstimator V3

V3 made the backward temporary factor optimizer-specific:

- AdamW: 0.65
- Adam: 0.65
- SGD: 0.35
- SGD with momentum: 0.45

V3 improved gpt2 SGD but damaged distilgpt2 SGD.

### TrainingMemoryEstimator V4

V4 made the SGD correction both optimizer-specific and model-size-aware:

- AdamW: 0.65
- Adam: 0.65
- SGD below 100M parameters: 0.65
- SGD above/equal 100M parameters: 0.35
- SGD with momentum: 0.45
- default: 0.50

This preserved distilgpt2 behavior while improving gpt2 SGD prediction.

## 7. V2 vs V3 vs V4 Comparison

| estimator                 | role                                                                                  |   num_rows |   allocated_MRE |   allocated_mean_error |   allocated_max_error |   reserved_MRE |   reserved_mean_error |   reserved_max_error |
|:--------------------------|:--------------------------------------------------------------------------------------|-----------:|----------------:|-----------------------:|----------------------:|---------------:|----------------------:|---------------------:|
| TrainingMemoryEstimatorV2 | V2: baseline training estimator with backward temporary memory correction             |         14 |         2.98779 |                4.95277 |               17.6936 |        3.68875 |               5.00371 |             16.8337  |
| TrainingMemoryEstimatorV3 | V3: optimizer-specific correction experiment                                          |         14 |         2.98779 |                4.36313 |               12.7747 |        4.20354 |               4.99495 |             12.5199  |
| TrainingMemoryEstimatorV4 | V4: candidate final estimator with optimizer-specific and model-size-aware correction |         14 |         2.97717 |                3.66309 |                8.768  |        3.68875 |               4.29286 |              9.52497 |

## 8. Final V4 Metrics

| metric               |   value_percent | meaning                                             |
|:---------------------|----------------:|:----------------------------------------------------|
| allocated_MRE        |         2.97717 | Median relative error for peak allocated memory     |
| allocated_mean_error |         3.66309 | Mean relative error for peak allocated memory       |
| allocated_max_error  |         8.768   | Worst-case relative error for peak allocated memory |
| reserved_MRE         |         3.68875 | Median relative error for peak reserved memory      |
| reserved_mean_error  |         4.29286 | Mean relative error for peak reserved memory        |
| reserved_max_error   |         9.52497 | Worst-case relative error for peak reserved memory  |

## 9. V4 Error by Model

| model_name   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| distilgpt2   |          8 |                3.46251 |                  2.97717 |               8.768   |               3.86821 |                 3.49684 |              9.52497 |                   1529.01 |                      1519.86 |                  1645.5  |                     1641.45 |
| gpt2         |          6 |                3.93054 |                  3.64661 |               6.67409 |               4.85907 |                 5.26279 |              6.91559 |                   1846.37 |                      1823.23 |                  2022.67 |                     1969.09 |

## 10. V4 Error by Optimizer

| optimizer_name   |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-----------------|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| adamw            |         10 |                3.39868 |                  2.96505 |               8.768   |               4.00727 |                 3.49684 |              9.52497 |                   1851.07 |                      1833.83 |                   2005.4 |                     1980.53 |
| sgd              |          4 |                4.32412 |                  4.44615 |               6.67409 |               5.00684 |                 5.21056 |              6.91559 |                   1199.9  |                      1190    |                   1311.5 |                     1285.2  |

## 11. V4 Error by Sequence Length

| model_name   |   input_tokens |   num_rows |   allocated_mean_error |   allocated_median_error |   allocated_max_error |   reserved_mean_error |   reserved_median_error |   reserved_max_error |   avg_actual_allocated_MB |   avg_predicted_allocated_MB |   avg_actual_reserved_MB |   avg_predicted_reserved_MB |
|:-------------|---------------:|-----------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------:|---------------------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|
| distilgpt2   |             32 |          1 |                4.98862 |                  4.98862 |               4.98862 |               3.88066 |                 3.88066 |              3.88066 |                   1592.43 |                      1512.99 |                     1700 |                     1634.03 |
| distilgpt2   |             64 |          6 |                3.61752 |                  2.97717 |               8.768   |               4.1476  |                 3.49684 |              9.52497 |                   1503.83 |                      1502.16 |                     1623 |                     1622.33 |
| distilgpt2   |            128 |          1 |                1.00629 |                  1.00629 |               1.00629 |               2.17944 |                 2.17944 |              2.17944 |                   1616.71 |                      1632.98 |                     1726 |                     1763.62 |
| gpt2         |             32 |          2 |                5.34591 |                  5.34591 |               5.91816 |               6.22601 |                 6.22601 |              6.91559 |                   1837.19 |                      1742.21 |                     2002 |                     1881.59 |
| gpt2         |             64 |          2 |                2.12482 |                  2.12482 |               2.51955 |               4.75779 |                 4.75779 |              4.98916 |                   1844.39 |                      1802.97 |                     2046 |                     1947.21 |
| gpt2         |            128 |          2 |                4.3209  |                  4.3209  |               6.67409 |               3.59341 |                 3.59341 |              5.89469 |                   1857.54 |                      1924.5  |                     2020 |                     2078.46 |

## 12. Key Findings

|   finding_id | finding                                                                 | evidence                                                                                                                                                              | importance                                                                    |
|-------------:|:------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
|            1 | TrainingMemoryEstimator V4 is the strongest current training estimator. | V4 achieved 2.98% allocated MRE, 3.66% allocated mean error, 3.69% reserved MRE, and 4.29% reserved mean error across combined distilgpt2 + gpt2 training validation. | This makes V4 the candidate final estimator for the training-memory phase.    |
|            2 | V4 fixed the distilgpt2 SGD failure caused by V3.                       | For distilgpt2 SGD at 64 tokens, V3 allocated error was 12.77%, while V4 restored it to 2.97%.                                                                        | This showed that optimizer-only correction was insufficient.                  |
|            3 | V4 preserved the gpt2 SGD improvement from V3.                          | For gpt2 SGD at 128 tokens, V2 allocated error was 17.69%, while V4 kept it at 6.67%.                                                                                 | This improved training-memory prediction for larger GPT-style models.         |
|            4 | Model-size-aware correction improved estimator balance.                 | V4 uses SGD factor 0.65 below 100M parameters and 0.35 above/equal 100M parameters.                                                                                   | This better handles different model scales.                                   |
|            5 | V4 reduced worst-case training prediction error.                        | Allocated max error dropped from 17.69% in V2 to 8.77% in V4; reserved max error dropped from 16.83% to 9.52%.                                                        | Lower worst-case error is important for deployment-style fit/fail prediction. |

## 13. Why V4 Is the Candidate Final Training Estimator

V4 solved the failure pattern discovered in V3.

V3 improved gpt2 SGD but damaged distilgpt2 SGD because it used the same SGD factor for all models.

V4 fixed this by making the SGD factor model-size-aware.

For distilgpt2 SGD at 64 tokens, V4 restored allocated error from 12.77% back to 2.97%.

For gpt2 SGD at 128 tokens, V4 preserved the improvement from V3: allocated error remained 6.67% instead of V2's 17.69%.

Globally, V4 reduced allocated mean error from 4.95% in V2 to 3.66%, and reserved mean error from 5.00% in V2 to 4.29%.

Therefore, V4 is the strongest current training estimator on the available validation set.

## 14. Limitations

- Validation currently covers distilgpt2 and gpt2 only.
- The 100M parameter threshold is empirical.
- The estimator is based on single-GPU Colab/T4-style measurements.
- Reserved memory depends on CUDA allocator behavior and may vary across runtime environments.
- Quantization, sparsity, and model-parallelism are not included in V4 yet.

## 15. Next Work

The next step is to evaluate V4 using PEF-style fit/fail simulation and safety-margin analysis.

After that, the project will add precision/quantization adaptation, sparsity adaptation, model-parallel memory partitioning, and CNN/Transformer architecture comparison.

## 16. Conclusion

TrainingMemoryEstimator V4 is the current candidate final estimator for the training-memory phase.

It combines optimizer-specific and model-size-aware backward temporary memory correction, improves gpt2 training prediction, preserves distilgpt2 performance, and reduces worst-case error.