
# Day 24 - distilgpt2 Training Batch-size Experiment

## Goal

Measure how distilgpt2 training memory changes with batch size.

Day 23 measured sequence-length scaling.

Day 24 measures batch-size scaling.

## Experiment Settings

- model = distilgpt2
- input_tokens = 64
- dtype = fp32
- optimizer = AdamW
- learning_rate = 5e-5

Batch sizes tested:

- 1
- 2
- 4

## Main Results

                 timestamp device_name gpu_name model_name     task  batch_size  input_tokens dtype optimizer_name  learning_rate   oom error_message     loss  runtime_sec  final_allocated_MB  final_reserved_MB  peak_allocated_MB  peak_reserved_MB                 experiment_type  fixed_input_tokens
2026-06-08T05:09:35.632732       colab Tesla T4 distilgpt2 training           1            64  fp32          adamw        0.00005 False               1.809547        9.727              972.10             1738.0            1600.19            1738.0 distilgpt2_batch_size_variation                  64
2026-06-08T05:09:46.701705       colab Tesla T4 distilgpt2 training           2            64  fp32          adamw        0.00005 False               1.878980        1.362              987.25             1726.0            1616.21            1726.0 distilgpt2_batch_size_variation                  64
2026-06-08T05:09:48.791799       colab Tesla T4 distilgpt2 training           4            64  fp32          adamw        0.00005 False               1.887785        1.362             1018.96             1768.0            1648.43            1768.0 distilgpt2_batch_size_variation                  64

## Batch Scaling Analysis

 batch_size  input_tokens  peak_allocated_MB  peak_reserved_MB  final_allocated_MB  final_reserved_MB  runtime_sec   oom  peak_allocated_per_sample_MB  peak_reserved_per_sample_MB
          1            64            1600.19            1738.0              972.10             1738.0        9.727 False                     1600.1900                       1738.0
          2            64            1616.21            1726.0              987.25             1726.0        1.362 False                      808.1050                        863.0
          4            64            1648.43            1768.0             1018.96             1768.0        1.362 False                      412.1075                        442.0

## Stage Peak Allocated Pivot

                   stage       1       2       4
          after_backward  956.49  972.51 1002.98
    after_batch_creation  313.24  331.74  331.74
           after_forward  382.29  454.84  571.78
              after_loss  382.29  454.84  571.78
        after_model_load  313.23  331.73  331.73
after_optimizer_creation  313.24  331.74  331.74
    after_optimizer_step 1600.19 1616.21 1648.43
         after_zero_grad 1600.19 1616.21 1648.43
       before_model_load    0.00   17.88   17.88

## Stage Peak Reserved Pivot

                   stage      1      2      4
          after_backward 1030.0 1038.0 1160.0
    after_batch_creation  350.0  370.0  370.0
           after_forward  416.0  462.0  616.0
              after_loss  416.0  462.0  616.0
        after_model_load  350.0  370.0  370.0
after_optimizer_creation  350.0  370.0  370.0
    after_optimizer_step 1738.0 1726.0 1768.0
         after_zero_grad 1738.0 1726.0 1768.0
       before_model_load    0.0   42.0   42.0

## Training vs Inference Batch Comparison

model_name  batch_size  input_tokens  training_peak_allocated_MB  training_peak_reserved_MB  inference_peak_allocated_MB  inference_peak_reserved_MB  allocated_training_vs_inference_ratio  reserved_training_vs_inference_ratio
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           1            64                     1600.19                     1738.0                       336.48                       368.0                               4.755676                              4.722826
distilgpt2           1            64                     1600.19                     1738.0                       341.35                       372.0                               4.687828                              4.672043
distilgpt2           1            64                     1600.19                     1738.0                       336.48                       368.0                               4.755676                              4.722826
distilgpt2           1            64                     1600.19                     1738.0                       336.48                       368.0                               4.755676                              4.722826
distilgpt2           1            64                     1600.19                     1738.0                       336.48                       368.0                               4.755676                              4.722826
distilgpt2           1            64                     1600.19                     1738.0                       341.50                       384.0                               4.685769                              4.526042
distilgpt2           1            64                     1600.19                     1738.0                       341.50                       384.0                               4.685769                              4.526042
distilgpt2           1            64                     1600.19                     1738.0                       341.50                       384.0                               4.685769                              4.526042
distilgpt2           1            64                     1600.19                     1738.0                       335.36                       364.0                               4.771559                              4.774725
distilgpt2           2            64                     1616.21                     1726.0                       343.24                       372.0                               4.708688                              4.639785
distilgpt2           4            64                     1648.43                     1768.0                       354.49                       396.0                               4.650145                              4.464646

## Summary

                    experiment  num_runs  min_batch_size  max_batch_size  min_peak_allocated_MB  max_peak_allocated_MB  min_peak_reserved_MB  max_peak_reserved_MB  oom_count
distilgpt2_training_batch_size         3               1               4                1600.19                1648.43                1726.0                1768.0          0

## Main Interpretation

This experiment measures how training memory changes with batch size.

Peak memory is expected to increase with batch size because larger batches require more activations and larger input tensors.

However, the increase may not be perfectly linear because parameters and optimizer states are shared across all samples in the batch.

## Next Step

Run optimizer comparison for distilgpt2: SGD vs AdamW.

