
# Day 25 - distilgpt2 Optimizer Comparison

## Goal

Compare training memory between SGD and AdamW for distilgpt2.

Day 23 and Day 24 showed that peak memory occurs after optimizer_step.

Day 25 isolates optimizer choice as the changed variable.

## Experiment Settings

- model = distilgpt2
- batch_size = 1
- input_tokens = 64
- dtype = fp32
- learning_rate = 5e-5

Optimizers tested:

- SGD
- AdamW

## Main Results

                 timestamp device_name gpu_name model_name     task  batch_size  input_tokens dtype optimizer_name  learning_rate   oom error_message     loss  runtime_sec  final_allocated_MB  final_reserved_MB  peak_allocated_MB  peak_reserved_MB                 experiment_type
2026-06-08T05:18:53.351795       colab Tesla T4 distilgpt2 training           1            64  fp32            sgd        0.00005 False               1.804632        9.183              345.63             1030.0             956.49            1030.0 distilgpt2_optimizer_comparison
2026-06-08T05:19:03.809840       colab Tesla T4 distilgpt2 training           1            64  fp32          adamw        0.00005 False               1.836824        1.441              973.48             1738.0            1600.94            1738.0 distilgpt2_optimizer_comparison

## Optimizer Memory Difference

            metric     sgd   adamw  difference_MB  percent_increase
 peak_allocated_MB  956.49 1600.94         644.45         67.376554
  peak_reserved_MB 1030.00 1738.00         708.00         68.737864
final_allocated_MB  345.63  973.48         627.85        181.653792
 final_reserved_MB 1030.00 1738.00         708.00         68.737864

## Optimizer Step Jump

optimizer_name  allocated_jump_backward_to_step_MB  reserved_jump_backward_to_step_MB  peak_allocated_jump_backward_to_step_MB  peak_reserved_jump_backward_to_step_MB  allocated_after_zero_grad_MB  reserved_after_zero_grad_MB
           sgd                                0.00                                0.0                                     0.00                                     0.0                        345.63                       1030.0
         adamw                              627.22                              688.0                                   644.45                                   688.0                        973.48                       1738.0

## Peak Allocated Pivot

                   stage   adamw    sgd
          after_backward  956.49 956.49
    after_batch_creation  331.74 313.24
           after_forward  391.04 382.29
              after_loss  391.04 382.29
        after_model_load  331.73 313.23
after_optimizer_creation  331.74 313.24
    after_optimizer_step 1600.94 956.49
         after_zero_grad 1600.94 956.49
       before_model_load   17.88   0.00

## Peak Reserved Pivot

                   stage  adamw    sgd
          after_backward 1050.0 1030.0
    after_batch_creation  370.0  350.0
           after_forward  436.0  416.0
              after_loss  436.0  416.0
        after_model_load  370.0  350.0
after_optimizer_creation  370.0  350.0
    after_optimizer_step 1738.0 1030.0
         after_zero_grad 1738.0 1030.0
       before_model_load   42.0    0.0

## Current Allocated Pivot

                   stage   adamw    sgd
          after_backward  660.49 660.49
    after_batch_creation  331.74 313.24
           after_forward  391.04 382.29
              after_loss  391.04 382.29
        after_model_load  331.73 313.23
after_optimizer_creation  331.74 313.24
    after_optimizer_step 1287.71 660.49
         after_zero_grad  973.48 345.63
       before_model_load   17.88   0.00

## Current Reserved Pivot

                   stage  adamw    sgd
          after_backward 1050.0 1030.0
    after_batch_creation  370.0  350.0
           after_forward  436.0  416.0
              after_loss  436.0  416.0
        after_model_load  370.0  350.0
after_optimizer_creation  370.0  350.0
    after_optimizer_step 1738.0 1030.0
         after_zero_grad 1738.0 1030.0
       before_model_load   42.0    0.0

## Summary

                     experiment  num_runs optimizers_tested  min_peak_allocated_MB  max_peak_allocated_MB  min_peak_reserved_MB  max_peak_reserved_MB  oom_count
distilgpt2_optimizer_comparison         2        sgd, adamw                 956.49                1600.94                1030.0                1738.0          0

## Main Interpretation

AdamW is expected to use more memory than SGD because AdamW stores optimizer states such as first and second moment tensors.

The most important comparison is after_optimizer_step and after_zero_grad because optimizer states become visible there.

This experiment supports the later OptimizerStateEstimator module.

## Next Step

Build OptimizerStateEstimator and TrainingMemoryEstimator.

