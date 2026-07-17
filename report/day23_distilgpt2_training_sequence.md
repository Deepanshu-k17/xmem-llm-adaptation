
# Day 23 - distilgpt2 Training Sequence-length Experiment

## Goal

Run the first meaningful training-memory experiment using distilgpt2.

Day 21 and Day 22 tested the training logger on tiny-gpt2.

Day 23 moves to distilgpt2 to observe more realistic training memory behavior.

## Experiment Settings

- model = distilgpt2
- batch_size = 1
- dtype = fp32
- optimizer = AdamW
- learning_rate = 5e-5

Input token lengths tested:

- 32
- 64
- 128

## Main Results

                 timestamp device_name gpu_name model_name     task  batch_size  input_tokens dtype optimizer_name  learning_rate   oom error_message     loss  runtime_sec  final_allocated_MB  final_reserved_MB  peak_allocated_MB  peak_reserved_MB                      experiment_type  sequence_length
2026-06-08T04:57:04.443752       colab Tesla T4 distilgpt2 training           1            32  fp32          adamw        0.00005 False               3.815424        9.323              965.34             1700.0            1592.43            1700.0 distilgpt2_sequence_length_variation               32
2026-06-08T04:57:15.648185       colab Tesla T4 distilgpt2 training           1            64  fp32          adamw        0.00005 False               2.154191        1.600              972.73             1738.0            1600.69            1738.0 distilgpt2_sequence_length_variation               64
2026-06-08T04:57:18.059131       colab Tesla T4 distilgpt2 training           1           128  fp32          adamw        0.00005 False               0.969172        1.549              987.62             1726.0            1616.71            1726.0 distilgpt2_sequence_length_variation              128

## Scaling Analysis

 input_tokens  peak_allocated_MB  peak_reserved_MB  final_allocated_MB  final_reserved_MB  runtime_sec   oom  allocated_per_token_MB  reserved_per_token_MB
           32            1592.43            1700.0              965.34             1700.0        9.323 False               49.763438              53.125000
           64            1600.69            1738.0              972.73             1738.0        1.600 False               25.010781              27.156250
          128            1616.71            1726.0              987.62             1726.0        1.549 False               12.630547              13.484375

## Stage Peak Allocated Pivot

                   stage      32      64     128
          after_backward  948.23  956.99  972.64
    after_batch_creation  313.24  331.74  331.74
           after_forward  352.65  391.04  456.09
              after_loss  352.65  391.04  456.09
        after_model_load  313.23  331.73  331.73
after_optimizer_creation  313.24  331.74  331.74
    after_optimizer_step 1592.43 1600.69 1616.71
         after_zero_grad 1592.43 1600.69 1616.71
       before_model_load    0.00   17.88   17.88

## Stage Peak Reserved Pivot

                   stage     32     64    128
          after_backward 1012.0 1030.0 1038.0
    after_batch_creation  350.0  350.0  350.0
           after_forward  388.0  416.0  462.0
              after_loss  388.0  416.0  462.0
        after_model_load  350.0  350.0  350.0
after_optimizer_creation  350.0  350.0  350.0
    after_optimizer_step 1700.0 1738.0 1726.0
         after_zero_grad 1700.0 1738.0 1726.0
       before_model_load    0.0   42.0   42.0

## Training vs Inference Comparison

model_name  input_tokens  training_peak_allocated_MB  training_peak_reserved_MB  inference_peak_allocated_MB  inference_peak_reserved_MB  allocated_training_vs_inference_ratio  reserved_training_vs_inference_ratio
distilgpt2            32                     1592.43                     1700.0                       332.18                       360.0                               4.793877                              4.722222
distilgpt2            64                     1600.69                     1738.0                       335.36                       364.0                               4.773050                              4.774725
distilgpt2            64                     1600.69                     1738.0                       335.36                       364.0                               4.773050                              4.774725
distilgpt2           128                     1616.71                     1726.0                       343.24                       366.0                               4.710145                              4.715847

## Summary

                         experiment  num_runs  min_input_tokens  max_input_tokens  min_peak_allocated_MB  max_peak_allocated_MB  min_peak_reserved_MB  max_peak_reserved_MB  oom_count
distilgpt2_training_sequence_length         3                32               128                1592.43                1616.71                1700.0                1738.0          0

## Main Interpretation

This experiment measures how distilgpt2 training memory changes with input sequence length.

Training memory is expected to increase with sequence length because longer sequences produce more activations that must be stored for backward computation.

The backward stage is expected to create the highest peak memory because it computes gradients and uses temporary tensors.

## Next Step

Run distilgpt2 training batch-size variation.

