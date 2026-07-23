
# Day 22 - tiny-gpt2 Training Sanity Experiments

## Goal

Run controlled tiny-gpt2 training experiments to verify the training memory logger.

Day 21 tested one training run.

Day 22 checks whether the logger behaves consistently under sequence-length variation and optimizer variation.

## Experiments Run

### 1. Sequence-length variation

Model:

sshleifer/tiny-gpt2

Fixed settings:

- batch_size = 1
- dtype = fp32
- optimizer = AdamW

Input token lengths tested:

- 16
- 32
- 64
- 128

### 2. Optimizer comparison

Model:

sshleifer/tiny-gpt2

Fixed settings:

- batch_size = 1
- input_tokens = 64
- dtype = fp32

Optimizers tested:

- SGD
- AdamW

## Sequence-length Results

                 timestamp device_name gpu_name          model_name     task  batch_size  input_tokens dtype optimizer_name  learning_rate   oom error_message      loss  runtime_sec  final_allocated_MB  final_reserved_MB  peak_allocated_MB  peak_reserved_MB           experiment_type  sequence_length
2026-06-04T04:57:20.820568       colab Tesla T4 sshleifer/tiny-gpt2 training           1            16  fp32          adamw        0.00005 False               10.828448        5.676               25.37               48.0              27.27              48.0 sequence_length_variation               16
2026-06-04T04:57:27.794366       colab Tesla T4 sshleifer/tiny-gpt2 training           1            32  fp32          adamw        0.00005 False               10.829083        1.060               28.44               68.0              45.29              68.0 sequence_length_variation               32
2026-06-04T04:57:29.319076       colab Tesla T4 sshleifer/tiny-gpt2 training           1            64  fp32          adamw        0.00005 False               10.830624        1.043               34.57               84.0              69.84              84.0 sequence_length_variation               64
2026-06-04T04:57:30.906285       colab Tesla T4 sshleifer/tiny-gpt2 training           1           128  fp32          adamw        0.00005 False               10.827228        1.057               46.84              132.0             118.95             132.0 sequence_length_variation              128

## Optimizer Results

                 timestamp device_name gpu_name          model_name     task  batch_size  input_tokens dtype optimizer_name  learning_rate   oom error_message      loss  runtime_sec  final_allocated_MB  final_reserved_MB  peak_allocated_MB  peak_reserved_MB      experiment_type
2026-06-04T04:58:15.812054       colab Tesla T4 sshleifer/tiny-gpt2 training           1            64  fp32            sgd        0.00005 False               10.831019        1.166               32.99               82.0              69.84              82.0 optimizer_comparison
2026-06-04T04:58:17.457428       colab Tesla T4 sshleifer/tiny-gpt2 training           1            64  fp32          adamw        0.00005 False               10.830429        1.035               34.57               84.0              69.84              84.0 optimizer_comparison

## Summary

               experiment  num_runs  min_peak_allocated_MB  max_peak_allocated_MB  min_peak_reserved_MB  max_peak_reserved_MB  oom_count
sequence_length_variation         4                  27.27                 118.95                  48.0                 132.0          0
     optimizer_comparison         2                  69.84                  69.84                  82.0                  84.0          0

## Optimizer Stage Pivot

                   stage  adamw   sgd
          after_backward  69.84 69.84
    after_batch_creation  20.72 20.72
           after_forward  45.30 45.30
              after_loss  45.30 45.30
        after_model_load  20.72 20.72
after_optimizer_creation  20.72 20.72
    after_optimizer_step  69.84 69.84
         after_zero_grad  69.84 69.84
       before_model_load  17.93 17.93

## Main Interpretation

The goal was not to get large memory values.

The goal was to confirm that the training logger is stable and can capture memory changes under controlled training settings.

tiny-gpt2 remains very small, so trends may be less clear than they will be for distilgpt2.

## Next Step

Move to distilgpt2 training sequence-length experiments.

