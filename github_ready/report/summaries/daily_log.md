
# Day 1 - Colab-Based LLM Inference Memory Logger Setup

## Goal

Set up the basic infrastructure for the internship project: a reproducible GPU memory profiling pipeline for LLM inference experiments.

The focus was not on running many experiments, but on building a clean logger that can record memory behavior consistently.

## Platform Used

- Platform: Google Colab
- GPU: Tesla T4
- Model used for first test: sshleifer/tiny-gpt2
- Mode: Inference

## Folder Structure Created

Created a Google Drive project folder:

MyDrive/
    xmem-llm-adaptation/
        notebooks/
        results/
        plots/
        report/
        src/
        src/estimators/

Purpose of this structure:

notebooks/  -> Colab experiment notebooks
results/    -> CSV files from experiments
plots/      -> generated plots
report/     -> daily logs and final notes
src/        -> reusable Python code
estimators/ -> future memory prediction modules

## Why Google Drive Was Used

Colab runtime storage is temporary. If the runtime disconnects or resets, local files can disappear.

Therefore, experiment results are saved directly to Google Drive so that:

1. results are not lost after runtime reset
2. experiments can be resumed later
3. CSV data can be used for plots and analysis
4. MRE/PEF can be calculated later
5. results remain reproducible and trackable

## Code Built

Created the first inference memory logger.

Main functions:

- clear_gpu_memory()
- get_memory_stats()
- append_to_csv()
- make_input_ids()
- run_inference_experiment()

## What Each Function Does

### 1. clear_gpu_memory()

Clears unused cached GPU memory and resets peak memory statistics.

Purpose:

Before each experiment, old memory peaks should not affect the current run.

### 2. get_memory_stats()

Records GPU memory statistics:

- peak_allocated_MB
- peak_reserved_MB
- final_allocated_MB
- final_reserved_MB

### 3. append_to_csv()

Saves each experiment result as one row in a CSV file.

CSV path:

/content/drive/MyDrive/xmem-llm-adaptation/results/inference_runs.csv

### 4. make_input_ids()

Creates input token sequences of exact length.

This is better than using text like:

"Hello " * input_tokens

because that does not guarantee exact token count after tokenization.

### 5. run_inference_experiment()

Runs one LLM inference experiment and records:

- timestamp
- platform
- gpu_name
- model_name
- mode
- batch_size
- input_tokens
- max_new_tokens
- output_tokens
- dtype
- use_cache
- peak_allocated_MB
- peak_reserved_MB
- final_allocated_MB
- final_reserved_MB
- runtime_sec
- oom
- error_message

## Important Concepts Learned

### Allocated Memory

Allocated memory means GPU memory currently used by live tensors.

Examples:

- model weights
- input tensors
- activations
- KV cache
- temporary tensors

### Reserved Memory

Reserved memory means GPU memory requested and held by PyTorch's CUDA caching allocator.

Reserved memory can be greater than allocated memory because PyTorch keeps memory blocks for reuse instead of immediately returning them to the GPU.

### Peak Memory

Peak memory means the maximum memory reached during the experiment.

This matters because GPU out-of-memory errors depend on the maximum memory spike, not the final memory after execution.

Example:

If peak memory is 10 GB but final memory is 4 GB, a 6 GB GPU can still fail.

### Why Exact Token Count Matters

LLM memory depends strongly on sequence length.

If we say input_tokens = 128, but the tokenizer actually creates 95 or 150 tokens, then memory analysis becomes incorrect.

Exact token count helps make experiments controlled and useful for prediction.

### use_cache=True

use_cache=True means the model stores past key and value tensors during generation.

This is called KV cache.

Effect:

- use_cache=True -> faster generation but extra memory usage
- use_cache=False -> slower generation but different/lower cache memory behavior

KV cache is important because it grows with:

- batch size
- sequence length
- number of layers
- hidden size
- dtype

## First Experiment Run

Configuration:

- model_name = sshleifer/tiny-gpt2
- batch_size = 1
- input_tokens = 32
- max_new_tokens = 32
- dtype = fp32
- use_cache = True
- GPU = Tesla T4

Result:

- output_tokens = 64
- peak_allocated_MB = 12.50
- peak_reserved_MB = 26.00
- final_allocated_MB = 11.91
- final_reserved_MB = 26.00
- runtime_sec = 0.872
- oom = False

## Interpretation of First Run

The model received 32 input tokens and generated 32 new tokens.

So:

output_tokens = input_tokens + max_new_tokens
output_tokens = 32 + 32 = 64

Peak allocated memory was 12.50 MB.

Final allocated memory was 11.91 MB.

This shows that memory peaked slightly during generation and then reduced after temporary tensors were freed.

Peak reserved memory and final reserved memory both stayed at 26.00 MB.

This shows PyTorch's caching allocator behavior: memory was reserved and kept for reuse.

## Day 1 Conclusion

On Day 1, I successfully built the basic Colab-based memory profiling pipeline and saved the first inference experiment result to CSV.

The main learning was the difference between:

- allocated memory -> active tensor memory
- reserved memory -> memory held by PyTorch allocator
- peak memory -> maximum memory during execution
- final memory -> memory remaining after execution

This setup is the foundation for future controlled experiments and memory prediction modules.

---


# Day 2 - Input Token Length vs Inference Memory

## Goal

Study how LLM inference memory changes when input sequence length increases.

This was a controlled experiment where only one variable was changed:

input_tokens

All other variables were kept constant.

## Fixed Variables

- model_name = sshleifer/tiny-gpt2
- batch_size = 1
- max_new_tokens = 32
- dtype = fp32
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

input_tokens = 16, 32, 64, 128, 256, 512

## Why Only Input Tokens Were Changed

Only one variable was changed so that the effect of input sequence length could be isolated.

If both input_tokens and max_new_tokens were changed together, it would be unclear whether memory changed because of input length, generated length, KV cache growth, or another factor.

This is why max_new_tokens was fixed at 32.

This is called a controlled experiment.

## Experiment Results

| input_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 48 | 12.50 | 26.00 | 11.91 | 26.00 | 0.126 | False |
| 32 | 64 | 12.50 | 26.00 | 11.91 | 26.00 | 0.086 | False |
| 64 | 96 | 12.50 | 26.00 | 11.92 | 26.00 | 0.085 | False |
| 128 | 160 | 12.51 | 26.00 | 11.92 | 26.00 | 0.133 | False |
| 256 | 288 | 12.51 | 26.00 | 11.92 | 26.00 | 0.144 | False |
| 512 | 544 | 12.53 | 26.00 | 11.93 | 26.00 | 0.090 | False |

## Observations

### 1. Peak allocated memory increased very slightly

Peak allocated memory changed from 12.50 MB at 16 input tokens to 12.53 MB at 512 input tokens.

The total increase was only 0.03 MB, so the graph is almost flat.

### 2. Peak reserved memory stayed constant

Peak reserved memory stayed fixed at 26.00 MB for all input token lengths.

This suggests that PyTorch reserved a memory block large enough for all these small runs and reused it without requesting more memory from the GPU.

### 3. Final allocated memory also changed very slightly

Final allocated memory changed from 11.91 MB to 11.93 MB.

This again shows that the model is too small for strong memory variation to appear.

### 4. Output tokens increased as expected

Since max_new_tokens = 32:

output_tokens = input_tokens + 32

Examples:

input_tokens = 16  -> output_tokens = 48
input_tokens = 512 -> output_tokens = 544

## Main Interpretation

The memory trend was almost flat because sshleifer/tiny-gpt2 is extremely small.

The model weight memory dominates the total memory footprint, and the sequence-length-dependent memory is too small to produce a strong visible trend.

Important correction:

Parameter size did not change.

The model remained the same in every run.

Only input sequence length changed.

The small memory increase comes from input-length-dependent computation, temporary tensors, activations, and KV-cache-related memory, not from model parameters.

## Why Tiny GPT-2 Is Not Enough

sshleifer/tiny-gpt2 is useful for checking whether the profiling pipeline works, but it is too small to prove strong LLM memory scaling.

Reasons:

1. very few parameters
2. small hidden size
3. small number of layers
4. small KV cache
5. very low memory usage overall

Larger models like distilgpt2, gpt2, facebook/opt-125m, or EleutherAI/gpt-neo-125M should show clearer trends.

## What This Experiment Shows

This experiment shows two important things.

First, the logging pipeline works correctly across multiple runs.

Second, for very small models, sequence length does not create a large memory difference.

This is still useful because it tells us that we need larger models for stronger memory-scaling analysis.

## Plots Created

Two plots were created and saved:

- plots/day2_peak_allocated_vs_input_tokens.png
- plots/day2_peak_reserved_vs_input_tokens.png

## Questions Answered

### Q1. Did peak_allocated_MB increase as input_tokens increased?

Yes, but only very slightly.

It increased from 12.50 MB at 16 input tokens to 12.53 MB at 512 input tokens.

The total increase was only 0.03 MB.

### Q2. Did peak_reserved_MB follow the same pattern as peak_allocated_MB?

No.

Peak reserved memory stayed constant at 26.00 MB for all input lengths.

This shows allocator reuse.

### Q3. Why did we keep max_new_tokens = 32 fixed?

We kept max_new_tokens fixed because we wanted to isolate the effect of input_tokens only.

If both input_tokens and max_new_tokens changed together, we would not know which variable caused the memory change.

### Q4. Why is sshleifer/tiny-gpt2 not enough to prove strong LLM memory scaling?

Because tiny-gpt2 is extremely small.

Its parameter count, hidden size, number of layers, and KV-cache size are tiny, so memory changes from sequence length are almost invisible.

Larger models are needed to observe clearer memory scaling.

### Q5. What variable should be tested next to study KV-cache behavior?

The next variable should be max_new_tokens.

Changing max_new_tokens changes the number of generated tokens, which is directly related to KV-cache memory growth.

## Day 2 Conclusion

On Day 2, I ran a controlled input-length experiment using sshleifer/tiny-gpt2.

Input tokens were varied from 16 to 512 while all other variables were fixed.

Peak allocated memory increased only slightly from 12.50 MB to 12.53 MB, while peak reserved memory stayed constant at 26.00 MB.

This suggests that tiny-gpt2 is too small to show strong sequence-length memory scaling, and larger models will be needed for clearer LLM memory behavior.

The next step is to study the effect of generated token length by varying max_new_tokens, which is directly related to KV-cache memory growth.

---


# Day 3 - Generated Token Length / KV-Cache Experiment

## Goal

Study how LLM inference memory changes when generated token length increases.

This experiment focuses on max_new_tokens, which is directly related to autoregressive generation and KV-cache behavior.

## Fixed Variables

- model_name = sshleifer/tiny-gpt2
- batch_size = 1
- input_tokens = 64
- dtype = fp32
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

max_new_tokens = 8, 16, 32, 64, 128, 256

## Why max_new_tokens Was Changed

During autoregressive generation, LLMs generate tokens one by one.

When use_cache=True, the model stores previous key and value tensors in memory so that it does not recompute attention over all previous tokens again and again.

This stored memory is called KV cache.

Increasing max_new_tokens increases the total generated sequence length, so this experiment helps study KV-cache-related memory behavior.

## Expected Output Token Rule

Since input_tokens = 64:

output_tokens = input_tokens + max_new_tokens

Examples:

- max_new_tokens = 8   -> output_tokens = 72
- max_new_tokens = 16  -> output_tokens = 80
- max_new_tokens = 32  -> output_tokens = 96
- max_new_tokens = 64  -> output_tokens = 128
- max_new_tokens = 128 -> output_tokens = 192
- max_new_tokens = 256 -> output_tokens = 320

## Experiment Results

| input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 8 | 72 | 12.50 | 26.00 | 11.92 | 26.00 | 1.006 | False |
| 64 | 16 | 80 | 12.50 | 26.00 | 11.92 | 26.00 | 0.051 | False |
| 64 | 32 | 96 | 12.50 | 26.00 | 11.92 | 26.00 | 0.090 | False |
| 64 | 64 | 128 | 12.50 | 26.00 | 11.92 | 26.00 | 0.240 | False |
| 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.620 | False |
| 64 | 256 | 320 | 12.51 | 26.00 | 11.92 | 26.00 | 0.709 | False |

## Observations

### 1. Peak allocated memory changed only slightly

Peak allocated memory changed from 12.50 MB at max_new_tokens = 8 to 12.51 MB at max_new_tokens = 256.

The total increase was only 0.01 MB.

This means tiny-gpt2 is too small to show strong generated-token memory scaling.

### 2. Peak reserved memory stayed constant

Peak reserved memory stayed fixed at 26.00 MB for all generated token lengths.

This suggests that PyTorch's CUDA caching allocator reserved a memory block and reused it across all these small runs.

### 3. Runtime generally increased with generated tokens

Runtime generally increased as max_new_tokens increased, especially from 32 to 256 generated tokens.

However, the first run with max_new_tokens = 8 took 1.006 seconds, which is unusually high compared to later runs.

This is likely due to warm-up overhead, CUDA/kernel initialization, or first-run setup cost.

So the correct interpretation is:

Ignoring the first warm-up-like run, runtime generally increases as max_new_tokens increases.

### 4. Output tokens increased correctly

Since input_tokens was fixed at 64:

output_tokens = 64 + max_new_tokens

This matched the experiment results.

Examples:

- max_new_tokens = 8 -> output_tokens = 72
- max_new_tokens = 256 -> output_tokens = 320

## Main Interpretation

This experiment was designed to study generated-token scaling and KV-cache-related memory behavior.

The memory increase was almost flat because sshleifer/tiny-gpt2 is extremely small.

The model has very few layers, very small hidden size, and very small attention/KV tensors.

Since gradients are not used during inference, the small memory change comes mainly from tiny KV-cache and temporary inference tensors, not gradients.

## Why This Experiment Matters

Day 2 tested input token length.

Day 3 tested generated token length.

Generated token length is more directly connected to KV cache because during generation, each new token can add new key/value tensors stored for later attention steps when use_cache=True.

This is why max_new_tokens is an important variable for LLM memory prediction.

## Plots Created

Three plots were created and saved:

- plots/day3_peak_allocated_vs_max_new_tokens.png
- plots/day3_peak_reserved_vs_max_new_tokens.png
- plots/day3_runtime_vs_max_new_tokens.png

## Questions Answered

### Q1. Did peak_allocated_MB increase when max_new_tokens increased?

Yes, but only extremely slightly.

It changed from 12.50 MB at max_new_tokens = 8 to 12.51 MB at max_new_tokens = 256.

The increase was only 0.01 MB.

### Q2. Did peak_reserved_MB change or stay flat?

Peak reserved memory stayed flat at 26.00 MB for all runs.

This shows allocator reuse for this small model.

### Q3. Why is max_new_tokens more directly related to KV cache than input_tokens?

max_new_tokens controls the number of generated tokens.

During autoregressive generation, each generated token can add key/value tensors to the KV cache when use_cache=True.

So increasing max_new_tokens directly increases the amount of generated-token history that may be stored.

### Q4. Did runtime increase as max_new_tokens increased?

Runtime generally increased for larger max_new_tokens.

The run with max_new_tokens = 8 was unusually slow at 1.006 seconds, likely due to warm-up overhead.

After that, runtime increased from 0.051 seconds at 16 tokens to 0.709 seconds at 256 tokens.

### Q5. Why might tiny-gpt2 still fail to show strong KV-cache memory growth?

tiny-gpt2 is too small.

It has very few layers, small hidden size, and very small attention/KV tensors.

Therefore, even when max_new_tokens increases, the extra KV-cache memory is tiny and does not strongly affect total GPU memory.

## Day 3 Conclusion

On Day 3, I studied the effect of generated token length on inference memory using sshleifer/tiny-gpt2.

I fixed input_tokens = 64, batch_size = 1, dtype = fp32, and use_cache = True, while varying max_new_tokens from 8 to 256.

Peak allocated memory changed only slightly from 12.50 MB to 12.51 MB, while peak reserved memory stayed constant at 26.00 MB.

This indicates that tiny-gpt2 is too small to show strong KV-cache memory scaling.

However, output token count increased correctly from 72 to 320, confirming that the experiment setup worked.

Runtime generally increased as max_new_tokens increased, although the first run showed unusually high runtime, likely due to warm-up overhead.

This experiment validates the generated-token profiling workflow, but larger models will be needed to observe clearer KV-cache memory behavior.

---


# Day 4 - use_cache True vs False Comparison

## Goal

Compare LLM inference memory and runtime when KV cache is enabled versus disabled.

This experiment directly studies the effect of use_cache, which controls whether the model stores past key/value tensors during autoregressive generation.

## Fixed Variables

- model_name = sshleifer/tiny-gpt2
- batch_size = 1
- input_tokens = 64
- max_new_tokens = 128
- dtype = fp32
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

- use_cache = True
- use_cache = False

Each setting was repeated 3 times because runtime can be noisy.

## Why use_cache Matters

During autoregressive generation, the model generates tokens one by one.

When use_cache=True, the model stores previous key/value tensors so it can avoid recomputing attention keys and values for all previous tokens again and again.

This stored memory is called KV cache.

Expected behavior in larger LLMs:

- use_cache=True usually makes generation faster
- use_cache=True can use extra memory because KV cache is stored
- use_cache=False may use less persistent cache memory
- use_cache=False can be slower because more computation is repeated

## Experiment Results

| use_cache | repeat_id | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| True | 1 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 1.367 | False |
| True | 2 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.550 | False |
| True | 3 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.382 | False |
| False | 1 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.562 | False |
| False | 2 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.373 | False |
| False | 3 | 64 | 128 | 192 | 12.50 | 26.00 | 11.92 | 26.00 | 0.382 | False |

## Summary Table

| use_cache | avg_peak_allocated_MB | avg_peak_reserved_MB | avg_runtime_sec | min_runtime_sec | max_runtime_sec |
|---|---:|---:|---:|---:|---:|
| False | 12.50 | 26.00 | 0.439 | 0.373 | 0.562 |
| True | 12.50 | 26.00 | 0.766 | 0.382 | 1.367 |

## Observations

### 1. Peak allocated memory did not change

Average peak allocated memory was:

- use_cache=True: 12.50 MB
- use_cache=False: 12.50 MB

So for tiny-gpt2, enabling or disabling cache did not show a measurable difference in peak allocated memory.

### 2. Peak reserved memory did not change

Average peak reserved memory was:

- use_cache=True: 26.00 MB
- use_cache=False: 26.00 MB

This means PyTorch's CUDA caching allocator reserved the same amount of memory in both cases.

### 3. Runtime was noisy

Average runtime was:

- use_cache=True: 0.766 seconds
- use_cache=False: 0.439 seconds

At first glance, this looks like use_cache=True was slower.

But this is not a reliable conclusion because the first use_cache=True run took 1.367 seconds, which is much higher than the next two runs.

This first run was likely affected by warm-up overhead, CUDA/kernel initialization, or first-run setup cost.

Ignoring the first use_cache=True run:

- use_cache=True average runtime becomes approximately 0.466 seconds
- use_cache=False average runtime remains approximately 0.439 seconds

So the runtime difference is small and noisy.

### 4. No strong cache effect was visible on tiny-gpt2

For this model, both memory and runtime differences were too small or noisy to draw strong conclusions about cache behavior.

This does not mean KV cache is unimportant.

It means tiny-gpt2 is too small to reveal meaningful KV-cache memory differences.

## Main Interpretation

The experiment successfully compared use_cache=True and use_cache=False under controlled settings.

However, tiny-gpt2 is too small to show strong memory differences between cache-enabled and cache-disabled generation.

Peak allocated memory stayed fixed at 12.50 MB.

Peak reserved memory stayed fixed at 26.00 MB.

Runtime was noisy, especially because the first use_cache=True run had a likely warm-up overhead.

The correct conclusion is:

The profiling pipeline works, but larger models are needed to observe meaningful KV-cache memory and runtime behavior.

## Why This Experiment Matters

This experiment completes the tiny-gpt2 baseline phase.

So far, I have tested:

- input token length
- generated token length
- use_cache=True vs use_cache=False

Across these experiments, tiny-gpt2 consistently showed almost flat memory behavior.

This tells us that tiny-gpt2 is useful for validating the pipeline, but not enough for strong LLM memory-scaling conclusions.

The next step is to move to a larger model such as distilgpt2.

## Questions Answered

### Q1. What was the average peak_allocated_MB for use_cache=True and use_cache=False?

- use_cache=True: 12.50 MB
- use_cache=False: 12.50 MB

There was no difference.

### Q2. What was the average runtime for use_cache=True and use_cache=False?

Including all runs:

- use_cache=True: 0.766 seconds
- use_cache=False: 0.439 seconds

But this is affected by the first use_cache=True run, which was unusually slow.

Ignoring the first use_cache=True run:

- use_cache=True average runtime is approximately 0.466 seconds
- use_cache=False average runtime is approximately 0.439 seconds

So the runtime difference is not strong.

### Q3. Did use_cache=True clearly use more memory in tiny-gpt2?

No.

Both cache settings used the same peak allocated memory and peak reserved memory.

### Q4. Why can use_cache=True make generation faster?

use_cache=True stores previous key/value tensors from earlier tokens.

This avoids recomputing attention information for all previous tokens during every new generation step.

So for larger models and longer sequences, use_cache=True usually improves generation speed.

### Q5. Why might the memory difference between use_cache=True and use_cache=False be small for tiny-gpt2?

Because tiny-gpt2 has:

- very few layers
- small hidden size
- tiny attention tensors
- tiny KV cache
- very low total memory use

So the KV-cache memory difference is too small to appear clearly in GPU memory measurements.

## Day 4 Conclusion

On Day 4, I compared use_cache=True and use_cache=False for tiny-gpt2 with input_tokens = 64 and max_new_tokens = 128.

Peak allocated memory stayed constant at 12.50 MB for all runs.

Peak reserved memory stayed constant at 26.00 MB for all runs.

This means tiny-gpt2 is too small to show a visible KV-cache memory difference.

Runtime was noisy. The first use_cache=True run was much slower, likely due to warm-up overhead.

After ignoring the first run, both cache settings had similar runtime.

Therefore, for tiny-gpt2, use_cache does not show a strong memory or speed difference.

The next step is to repeat core experiments on a larger model like distilgpt2.

---


# Day 5 - distilgpt2 Input Token Length Experiment

## Goal

Repeat the input-token-length memory experiment on a larger model: distilgpt2.

Days 2 to 4 showed that sshleifer/tiny-gpt2 is useful for validating the profiling pipeline, but it is too small to reveal strong memory-scaling behavior.

The goal of Day 5 was to check whether a larger Transformer model shows clearer memory growth with increasing input sequence length.

## Fixed Variables

- model_name = distilgpt2
- batch_size = 1
- max_new_tokens = 32
- dtype = fp32
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

input_tokens = 16, 32, 64, 128, 256, 512

## Why distilgpt2 Was Used

Earlier experiments with sshleifer/tiny-gpt2 showed almost flat memory behavior.

For tiny-gpt2:

- input_tokens increased from 16 to 512
- peak_allocated_MB increased only from 12.50 MB to 12.53 MB
- peak_reserved_MB stayed constant at 26.00 MB

This showed that tiny-gpt2 is too small to reveal meaningful memory scaling.

distilgpt2 was used because it is still small enough to run on free GPUs like Tesla T4, but it is much larger and more realistic than tiny-gpt2.

## Experiment Results

| model_name | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilgpt2 | 16 | 32 | 48 | 331.22 | 358.00 | 328.99 | 358.00 | 1.128 | False |
| distilgpt2 | 32 | 32 | 64 | 332.18 | 360.00 | 328.99 | 360.00 | 0.208 | False |
| distilgpt2 | 64 | 32 | 96 | 335.36 | 364.00 | 328.99 | 364.00 | 0.224 | False |
| distilgpt2 | 128 | 32 | 160 | 343.24 | 366.00 | 328.99 | 366.00 | 0.341 | False |
| distilgpt2 | 256 | 32 | 288 | 354.50 | 396.00 | 328.99 | 396.00 | 0.194 | False |
| distilgpt2 | 512 | 32 | 544 | 384.01 | 396.00 | 329.00 | 396.00 | 0.233 | False |

## Observations

### 1. distilgpt2 used much more memory than tiny-gpt2

At 512 input tokens:

- tiny-gpt2 peak allocated memory = 12.53 MB
- distilgpt2 peak allocated memory = 384.01 MB

Ratio:

384.01 / 12.53 = approximately 30.65

So, distilgpt2 used around 30.6 times more peak allocated memory than tiny-gpt2 at 512 input tokens.

This confirms that model size heavily affects GPU memory consumption.

### 2. Peak allocated memory clearly increased with input length

For distilgpt2:

- 16 input tokens -> 331.22 MB
- 512 input tokens -> 384.01 MB

Increase:

384.01 - 331.22 = 52.79 MB

Percentage increase:

52.79 / 331.22 * 100 = approximately 15.94%

So peak allocated memory increased by about 15.9% when input length increased from 16 to 512 tokens.

This is a meaningful trend, unlike tiny-gpt2 where the change was almost invisible.

### 3. Peak reserved memory increased and then plateaued

Peak reserved memory changed as follows:

- 16 tokens -> 358 MB
- 32 tokens -> 360 MB
- 64 tokens -> 364 MB
- 128 tokens -> 366 MB
- 256 tokens -> 396 MB
- 512 tokens -> 396 MB

Peak reserved memory increased from 358 MB to 396 MB, then stayed constant from 256 to 512 tokens.

This shows PyTorch CUDA caching allocator behavior.

The allocator requested larger memory blocks as input length increased, but after reserving enough memory, it reused the same reserved block for later runs.

### 4. Final allocated memory stayed almost constant

Final allocated memory stayed around 329 MB for all input lengths.

This means the model's persistent memory after generation was mostly constant.

The increase in peak allocated memory happened during generation because of temporary tensors, attention computation, activations, and KV-cache-related memory.

This shows why peak memory is more important than final memory for memory prediction.

### 5. Runtime was noisy

Runtime values did not increase smoothly.

The first run took 1.128 seconds, likely because of warm-up overhead, model setup, CUDA kernel initialization, or first-run effects.

Later runtimes were smaller and somewhat noisy.

Therefore, runtime should not be over-interpreted from this experiment.

The main useful result from Day 5 is memory scaling, not runtime scaling.

## Main Interpretation

Day 5 is the first experiment where memory scaling became clearly visible.

With tiny-gpt2, peak allocated memory was almost flat because the model was extremely small.

With distilgpt2, peak allocated memory increased from 331.22 MB to 384.01 MB as input tokens increased from 16 to 512.

This confirms that larger models show more meaningful memory behavior and are better for this internship project.

The experiment also shows that:

- model size creates a large base memory cost
- input length adds extra memory on top of model memory
- peak allocated memory captures temporary memory growth
- peak reserved memory reflects PyTorch allocator behavior
- final allocated memory alone is not enough to understand peak GPU pressure

## Comparison With tiny-gpt2

### tiny-gpt2 Day 2 result

At 512 input tokens:

- peak_allocated_MB = 12.53 MB
- peak_reserved_MB = 26.00 MB

### distilgpt2 Day 5 result

At 512 input tokens:

- peak_allocated_MB = 384.01 MB
- peak_reserved_MB = 396.00 MB

### Key comparison

distilgpt2 used about 30.6 times more peak allocated memory than tiny-gpt2 at 512 input tokens.

This proves that tiny-gpt2 was only useful for pipeline validation, while distilgpt2 gives more meaningful memory-scaling behavior.

## Why distilgpt2 Uses More Memory

distilgpt2 uses more memory because compared to sshleifer/tiny-gpt2, it has:

- far more parameters
- larger hidden dimensions
- larger attention tensors
- larger activations
- larger KV-cache tensors
- higher base model memory

So even before sequence-length effects appear, distilgpt2 already has a much larger memory footprint.

## Why This Experiment Is Useful

Even if the memory increase with input length is not extremely large, the experiment is still useful because:

1. it proves that model size strongly affects memory
2. it shows visible memory scaling with input length
3. it confirms that larger models are needed for meaningful analysis
4. it shows the difference between peak allocated and final allocated memory
5. it shows allocator behavior through peak reserved memory
6. it prepares the project for building memory prediction formulas later

## Plots Created

The following plots were created and saved:

- plots/day5_distilgpt2_peak_allocated_vs_input_tokens.png
- plots/day5_distilgpt2_peak_reserved_vs_input_tokens.png
- plots/day5_tiny_vs_distilgpt2_peak_allocated.png

## Questions Answered

### Q1. How much higher is distilgpt2 memory compared to tiny-gpt2?

At 512 input tokens:

- tiny-gpt2 peak allocated memory = 12.53 MB
- distilgpt2 peak allocated memory = 384.01 MB

distilgpt2 used approximately 30.6 times more peak allocated memory than tiny-gpt2.

### Q2. Did peak_allocated_MB increase with input length for distilgpt2?

Yes.

Peak allocated memory increased from 331.22 MB at 16 input tokens to 384.01 MB at 512 input tokens.

The increase was 52.79 MB, or about 15.9%.

### Q3. Did peak_reserved_MB stay flat or increase?

Peak reserved memory increased from 358 MB to 396 MB.

It increased as input length increased, then plateaued at 396 MB from 256 to 512 input tokens.

This shows PyTorch allocator behavior.

### Q4. Why is distilgpt2 expected to use more memory than tiny-gpt2?

distilgpt2 is much larger than sshleifer/tiny-gpt2.

It has far more parameters, larger hidden dimensions, larger attention tensors, larger activations, and larger KV-cache tensors.

Because of this, it has a much higher base memory requirement and shows clearer memory scaling.

### Q5. Even if memory does not increase much with input length, why is the experiment still useful?

The experiment is useful because it shows how memory changes when moving from a tiny test model to a more realistic Transformer model.

It also shows that model size dominates base memory, while input length adds extra memory.

This helps build intuition for future memory prediction modules.

## Day 5 Conclusion

On Day 5, I repeated the input-token-length experiment using distilgpt2 instead of sshleifer/tiny-gpt2.

The purpose was to check whether a larger Transformer model shows clearer memory scaling.

The result showed a clear increase in peak allocated memory as input length increased.

Peak allocated memory increased from 331.22 MB at 16 input tokens to 384.01 MB at 512 input tokens, an increase of 52.79 MB or about 15.9%.

At 512 input tokens, distilgpt2 used 384.01 MB peak allocated memory compared to only 12.53 MB for tiny-gpt2.

This means distilgpt2 used around 30.6 times more peak allocated memory.

Peak reserved memory increased from 358 MB to 396 MB and then plateaued from 256 to 512 tokens.

This shows PyTorch allocator behavior: it reserved larger memory blocks when required and then reused them.

This experiment confirms that tiny-gpt2 was only useful for validating the profiling pipeline, while distilgpt2 gives more meaningful memory-scaling behavior.

The next step is to repeat the generated-token-length experiment on distilgpt2 to study KV-cache behavior more clearly.

---


# Day 6 - distilgpt2 Generated Token Length / KV-Cache Experiment

## Goal

Repeat the generated-token-length experiment using distilgpt2 instead of sshleifer/tiny-gpt2.

Day 3 tested max_new_tokens on tiny-gpt2, but the memory change was almost invisible because the model was too small.

The goal of Day 6 was to check whether a larger model shows clearer memory scaling when generated token length increases.

This experiment is directly related to KV-cache behavior.

## Fixed Variables

- model_name = distilgpt2
- batch_size = 1
- input_tokens = 64
- dtype = fp32
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

max_new_tokens = 8, 16, 32, 64, 128, 256

## Why max_new_tokens Was Changed

During autoregressive generation, the model generates tokens one by one.

When use_cache=True, the model stores previous key/value tensors so that it does not recompute attention information for all previous tokens again and again.

This stored memory is called KV cache.

Increasing max_new_tokens increases the generated sequence length, so this experiment helps study generated-token and KV-cache-related memory behavior.

## Expected Output Token Rule

Since input_tokens = 64:

output_tokens = input_tokens + max_new_tokens

Examples:

- max_new_tokens = 8   -> output_tokens = 72
- max_new_tokens = 16  -> output_tokens = 80
- max_new_tokens = 32  -> output_tokens = 96
- max_new_tokens = 64  -> output_tokens = 128
- max_new_tokens = 128 -> output_tokens = 192
- max_new_tokens = 256 -> output_tokens = 320

## Experiment Results

| model_name | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilgpt2 | 64 | 8 | 72 | 335.36 | 364.00 | 328.99 | 364.00 | 0.915 | False |
| distilgpt2 | 64 | 16 | 80 | 335.36 | 364.00 | 328.99 | 364.00 | 0.089 | False |
| distilgpt2 | 64 | 32 | 96 | 335.36 | 364.00 | 328.99 | 364.00 | 0.239 | False |
| distilgpt2 | 64 | 64 | 128 | 335.36 | 364.00 | 328.99 | 364.00 | 0.613 | False |
| distilgpt2 | 64 | 128 | 192 | 336.48 | 368.00 | 328.99 | 368.00 | 1.015 | False |
| distilgpt2 | 64 | 256 | 320 | 341.35 | 372.00 | 328.99 | 372.00 | 1.554 | False |

## Observations

### 1. Peak allocated memory increased with generated tokens

Peak allocated memory stayed constant at 335.36 MB from max_new_tokens = 8 to 64.

Then it increased:

- max_new_tokens = 128 -> 336.48 MB
- max_new_tokens = 256 -> 341.35 MB

Overall:

- max_new_tokens = 8 -> 335.36 MB
- max_new_tokens = 256 -> 341.35 MB

Increase:

341.35 - 335.36 = 5.99 MB

Percentage increase:

5.99 / 335.36 * 100 = approximately 1.79%

So the increase is visible, but not huge.

### 2. Peak reserved memory increased in chunks

Peak reserved memory stayed constant at 364 MB from max_new_tokens = 8 to 64.

Then it increased:

- max_new_tokens = 128 -> 368 MB
- max_new_tokens = 256 -> 372 MB

This shows PyTorch CUDA caching allocator behavior.

Reserved memory does not always increase smoothly. It often grows in chunks when the allocator needs a larger block.

### 3. Final allocated memory stayed constant

Final allocated memory stayed around 328.99 MB for all runs.

This means the persistent memory after generation was almost constant.

The extra memory appeared mainly in peak memory, not final memory.

This again shows why peak memory is more important than final memory for GPU memory prediction.

### 4. Runtime generally increased with max_new_tokens

The first run with max_new_tokens = 8 took 0.915 seconds, which is unusually high.

This was likely due to warm-up overhead, CUDA/kernel initialization, or first-run setup.

Ignoring the first warm-up-like run, runtime increased clearly:

- max_new_tokens = 16 -> 0.089 seconds
- max_new_tokens = 32 -> 0.239 seconds
- max_new_tokens = 64 -> 0.613 seconds
- max_new_tokens = 128 -> 1.015 seconds
- max_new_tokens = 256 -> 1.554 seconds

This shows that longer generation requires more computation.

## Comparison With tiny-gpt2

At max_new_tokens = 256:

- tiny-gpt2 peak allocated memory = 12.51 MB
- distilgpt2 peak allocated memory = 341.35 MB

Ratio:

341.35 / 12.51 = approximately 27.29

So distilgpt2 used about 27.3 times more peak allocated memory than tiny-gpt2 for the same generated-token setting.

This confirms that tiny-gpt2 was too small for meaningful KV-cache memory analysis, while distilgpt2 shows more visible memory behavior.

## Main Interpretation

Day 6 is more useful than Day 3 because distilgpt2 is large enough to show visible generated-token memory scaling.

Peak allocated memory increased as max_new_tokens increased, but the increase was moderate because base model memory still dominates total GPU memory.

Peak reserved memory increased in chunks, showing allocator behavior.

Runtime increased clearly after the first warm-up-like run, showing that longer generation requires more computation.

This experiment supports the idea that generated-token length is an important variable for LLM memory prediction, especially because of KV-cache-related memory.

## Why This Experiment Matters for the Project

This project is about adapting xMem-style memory prediction to LLM workloads.

One LLM-specific memory component is KV cache.

KV-cache memory depends on:

- batch size
- total sequence length
- number of layers
- hidden size
- dtype
- use_cache setting

Day 6 provides experimental evidence that generated-token length affects memory more clearly when the model is larger.

This will later help build the KVCacheEstimator module.

## Plots Created

The following plots were created and saved:

- plots/day6_distilgpt2_peak_allocated_vs_max_new_tokens.png
- plots/day6_distilgpt2_peak_reserved_vs_max_new_tokens.png
- plots/day6_distilgpt2_runtime_vs_max_new_tokens.png
- plots/day6_tiny_vs_distilgpt2_generated_token_memory.png

## Questions Answered

### Q1. Did peak_allocated_MB increase with max_new_tokens?

Yes.

Peak allocated memory increased from 335.36 MB at max_new_tokens = 8 to 341.35 MB at max_new_tokens = 256.

The total increase was 5.99 MB, or about 1.79%.

### Q2. Did peak_reserved_MB increase smoothly, jump, or stay flat?

Peak reserved memory stayed flat at 364 MB from max_new_tokens = 8 to 64.

Then it jumped to 368 MB at max_new_tokens = 128 and 372 MB at max_new_tokens = 256.

So it increased in chunks, not smoothly.

### Q3. How much more memory did distilgpt2 use than tiny-gpt2 at max_new_tokens = 256?

At max_new_tokens = 256:

- tiny-gpt2 peak allocated memory = 12.51 MB
- distilgpt2 peak allocated memory = 341.35 MB

distilgpt2 used approximately 27.3 times more peak allocated memory.

### Q4. Did runtime generally increase as generated tokens increased?

Yes, after ignoring the first warm-up-like run.

The first run took 0.915 seconds, likely due to warm-up overhead.

After that, runtime increased from 0.089 seconds at 16 generated tokens to 1.554 seconds at 256 generated tokens.

### Q5. Why is this experiment more useful than Day 3?

Day 3 used tiny-gpt2, where memory changes were almost invisible.

Day 6 used distilgpt2, which is larger and shows visible changes in both peak allocated memory and peak reserved memory as generated tokens increase.

This makes Day 6 more useful for understanding KV-cache-related memory behavior.

## Day 6 Conclusion

On Day 6, I repeated the generated-token-length experiment using distilgpt2.

I fixed input_tokens = 64, batch_size = 1, dtype = fp32, and use_cache = True, while varying max_new_tokens from 8 to 256.

Peak allocated memory stayed constant at 335.36 MB from 8 to 64 generated tokens, then increased to 336.48 MB at 128 tokens and 341.35 MB at 256 tokens.

Overall, peak allocated memory increased by 5.99 MB, or about 1.79%.

Peak reserved memory stayed constant at 364 MB from 8 to 64 generated tokens, then increased to 368 MB at 128 tokens and 372 MB at 256 tokens.

This shows PyTorch allocator behavior, where reserved memory grows in chunks instead of smoothly.

Compared to tiny-gpt2, distilgpt2 used much more memory.

At max_new_tokens = 256, tiny-gpt2 used 12.51 MB peak allocated memory, while distilgpt2 used 341.35 MB.

This is about 27.3 times more memory.

Runtime generally increased as max_new_tokens increased after ignoring the first warm-up-like run.

This confirms that longer generation requires more computation and can also increase memory due to KV-cache-related storage.

This experiment is more useful than Day 3 because distilgpt2 is large enough to show visible generated-token memory scaling.

The next step is to compare use_cache=True and use_cache=False on distilgpt2.

---


# Day 7 - distilgpt2 use_cache True vs False Comparison

## Goal

Compare LLM inference memory and runtime when KV cache is enabled versus disabled on distilgpt2.

Day 4 tested use_cache=True vs use_cache=False on sshleifer/tiny-gpt2, but tiny-gpt2 was too small to show meaningful memory differences.

The goal of Day 7 was to check whether a larger model, distilgpt2, shows clearer cache-related memory behavior.

## Fixed Variables

- model_name = distilgpt2
- batch_size = 1
- input_tokens = 64
- max_new_tokens = 128
- dtype = fp32
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

- use_cache = True
- use_cache = False

Each setting was repeated 3 times because runtime can be noisy.

## Why use_cache Matters

During autoregressive generation, the model generates tokens one by one.

When use_cache=True, the model stores previous key/value tensors so it can avoid recomputing attention information for all previous tokens again and again.

This stored memory is called KV cache.

Expected behavior in larger LLMs:

- use_cache=True can make generation faster
- use_cache=True stores KV cache
- use_cache=False may avoid storing past KV cache in the same way
- use_cache=False can require repeated full-context computation
- memory behavior may not be obvious without profiling

## Experiment Results

| model_name | use_cache | repeat_id | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilgpt2 | True | 1 | 64 | 128 | 192 | 336.48 | 368.00 | 328.99 | 368.00 | 1.491 | False |
| distilgpt2 | True | 2 | 64 | 128 | 192 | 336.48 | 368.00 | 328.99 | 368.00 | 0.720 | False |
| distilgpt2 | True | 3 | 64 | 128 | 192 | 336.48 | 368.00 | 328.99 | 368.00 | 1.019 | False |
| distilgpt2 | False | 1 | 64 | 128 | 192 | 341.50 | 384.00 | 328.99 | 384.00 | 1.132 | False |
| distilgpt2 | False | 2 | 64 | 128 | 192 | 341.50 | 384.00 | 328.99 | 384.00 | 1.085 | False |
| distilgpt2 | False | 3 | 64 | 128 | 192 | 341.50 | 384.00 | 328.99 | 384.00 | 0.993 | False |

## Summary Table

| use_cache | avg_peak_allocated_MB | avg_peak_reserved_MB | avg_runtime_sec | min_runtime_sec | max_runtime_sec |
|---|---:|---:|---:|---:|---:|
| False | 341.50 | 384.00 | 1.070 | 0.993 | 1.132 |
| True | 336.48 | 368.00 | 1.077 | 0.720 | 1.491 |

## Observations

### 1. use_cache=False used more peak allocated memory

Average peak allocated memory:

- use_cache=True: 336.48 MB
- use_cache=False: 341.50 MB

Difference:

341.50 - 336.48 = 5.02 MB

So use_cache=False used 5.02 MB more peak allocated memory than use_cache=True.

This is an important result because it shows that use_cache=True does not always mean higher peak memory in practice.

### 2. use_cache=False also used more peak reserved memory

Average peak reserved memory:

- use_cache=True: 368.00 MB
- use_cache=False: 384.00 MB

Difference:

384.00 - 368.00 = 16.00 MB

So use_cache=False caused PyTorch to reserve 16 MB more memory.

This shows allocator-level behavior, not just active tensor memory.

### 3. Runtime was almost the same

Average runtime:

- use_cache=True: 1.077 seconds
- use_cache=False: 1.070 seconds

The difference is very small.

So this experiment does not show a clear runtime advantage for either cache setting.

Runtime is also noisy on Colab, so this result should not be over-interpreted.

### 4. Cache behavior is nuanced

A simple assumption would be:

use_cache=True stores KV cache, so it must always use more memory.

But the experiment showed the opposite:

use_cache=False used more peak allocated and peak reserved memory.

This can happen because use_cache=False may avoid persistent KV-cache storage, but it can require repeated full-context computation during generation, creating more temporary memory pressure.

So memory behavior depends on both persistent cache tensors and temporary computation tensors.

## Main Interpretation

Day 7 showed that cache behavior is not as simple as cache=True means more memory and cache=False means less memory.

For distilgpt2 with input_tokens = 64 and max_new_tokens = 128:

- use_cache=True had lower peak allocated memory
- use_cache=True had lower peak reserved memory
- runtime was almost the same for both settings

This means empirical profiling is necessary.

Theoretical assumptions alone are not enough to understand LLM memory behavior.

## Why This Experiment Matters for the Project

This project is about adapting xMem-style memory prediction to LLM workloads.

xMem-style prediction needs to account for:

- live tensor memory
- temporary computation memory
- allocator reserved memory
- sequence length
- generated length
- cache behavior

Day 7 provides evidence that cache behavior affects memory in a non-obvious way.

A future KV-cache estimator cannot simply assume that use_cache=True always increases peak memory.

It must consider that use_cache=False can create larger temporary memory pressure during repeated computation.

## Plots Created

The following plots were created and saved:

- plots/day7_distilgpt2_avg_peak_allocated_use_cache.png
- plots/day7_distilgpt2_avg_peak_reserved_use_cache.png
- plots/day7_distilgpt2_avg_runtime_use_cache.png
- plots/day7_distilgpt2_runtime_variability_use_cache.png

## Questions Answered

### Q1. What was the average peak_allocated_MB for use_cache=True and use_cache=False?

- use_cache=True: 336.48 MB
- use_cache=False: 341.50 MB

use_cache=False used 5.02 MB more peak allocated memory.

### Q2. What was the average peak_reserved_MB for use_cache=True and use_cache=False?

- use_cache=True: 368.00 MB
- use_cache=False: 384.00 MB

use_cache=False used 16 MB more peak reserved memory.

### Q3. What was the average runtime for use_cache=True and use_cache=False?

- use_cache=True: 1.077 seconds
- use_cache=False: 1.070 seconds

Runtime was almost identical.

### Q4. Did use_cache=True clearly improve runtime on distilgpt2?

No.

The average runtime was almost the same for both settings.

This experiment does not show a clear runtime advantage for use_cache=True.

### Q5. Did use_cache=True clearly increase memory?

No.

The opposite happened.

use_cache=False used more memory:

- 5.02 MB more peak allocated memory
- 16 MB more peak reserved memory

This may be because use_cache=False can require repeated full-context computation, which creates more temporary memory pressure.

## Day 7 Conclusion

On Day 7, I compared use_cache=True and use_cache=False for distilgpt2 with input_tokens = 64 and max_new_tokens = 128.

The result showed that use_cache=False used more memory than use_cache=True.

Average peak allocated memory was 336.48 MB for use_cache=True and 341.50 MB for use_cache=False, a difference of 5.02 MB.

Average peak reserved memory was 368 MB for use_cache=True and 384 MB for use_cache=False, a difference of 16 MB.

Runtime was almost the same for both settings: 1.077 seconds for use_cache=True and 1.070 seconds for use_cache=False.

Therefore, this experiment does not show a clear runtime advantage for either setting.

The main observation is that cache behavior is nuanced.

use_cache=True stores KV cache, but use_cache=False may require repeated full-context computation and can create more temporary memory pressure.

This shows why empirical profiling is necessary instead of assuming memory behavior from theory alone.

The next step is to study the effect of batch size on distilgpt2 memory usage.

---


# Day 8 - distilgpt2 Batch Size Memory Experiment

## Goal

Study how inference memory changes when batch size increases.

Batch size is one of the most important hyperparameters for memory prediction because it directly affects activations, attention tensors, KV-cache memory, input tensors, and temporary computation buffers.

The goal of Day 8 was to understand how total memory and memory per sample change as batch size increases.

## Fixed Variables

- model_name = distilgpt2
- input_tokens = 64
- max_new_tokens = 64
- dtype = fp32
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

batch_size = 1, 2, 4, 8

## Why Batch Size Was Changed

Batch size controls how many input samples are processed together in one forward/generation run.

Increasing batch size usually increases total memory because batch-dependent tensors become larger.

These include:

- input tensors
- activations
- attention tensors
- KV-cache tensors
- temporary buffers

However, model weights are shared across the whole batch.

This means memory does not usually scale perfectly linearly with batch size.

## Experiment Results

| model_name | batch_size | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilgpt2 | 1 | 64 | 64 | 128 | 335.36 | 364.00 | 328.99 | 364.00 | 1.243 | False |
| distilgpt2 | 2 | 64 | 64 | 128 | 343.24 | 372.00 | 328.99 | 372.00 | 0.796 | False |
| distilgpt2 | 4 | 64 | 64 | 128 | 354.49 | 396.00 | 328.99 | 396.00 | 0.631 | False |
| distilgpt2 | 8 | 64 | 64 | 128 | 384.00 | 416.00 | 329.00 | 416.00 | 0.437 | False |

## Efficiency Table

| batch_size | peak_allocated_MB | peak_allocated_per_sample_MB | runtime_sec | runtime_per_sample_sec |
|---:|---:|---:|---:|---:|
| 1 | 335.36 | 335.3600 | 1.243 | 1.243000 |
| 2 | 343.24 | 171.6200 | 0.796 | 0.398000 |
| 4 | 354.49 | 88.6225 | 0.631 | 0.157750 |
| 8 | 384.00 | 48.0000 | 0.437 | 0.054625 |

## Observations

### 1. Peak allocated memory increased with batch size

Peak allocated memory increased from:

- batch_size = 1 -> 335.36 MB
- batch_size = 8 -> 384.00 MB

Increase:

384.00 - 335.36 = 48.64 MB

Percentage increase:

48.64 / 335.36 * 100 = approximately 14.50%

So increasing batch size from 1 to 8 increased peak allocated memory by about 14.5%.

### 2. Memory scaling was sub-linear

Batch size increased 8 times, from 1 to 8.

If memory scaled perfectly linearly, batch_size = 8 would have used approximately:

335.36 * 8 = 2682.88 MB

But actual peak allocated memory was only:

384.00 MB

So memory scaling was strongly sub-linear.

This happened because model weights are loaded once and shared across all samples in the batch.

Only batch-dependent tensors grow with batch size.

### 3. Memory per sample decreased sharply

Peak allocated memory per sample decreased from:

- batch_size = 1 -> 335.36 MB/sample
- batch_size = 8 -> 48.00 MB/sample

This shows that larger batch sizes can be more memory-efficient per sample, even though total memory increases.

The fixed model memory gets amortized across more samples.

### 4. Peak reserved memory increased in chunks

Peak reserved memory changed as follows:

- batch_size = 1 -> 364 MB
- batch_size = 2 -> 372 MB
- batch_size = 4 -> 396 MB
- batch_size = 8 -> 416 MB

This increase was not perfectly linear.

It reflects PyTorch CUDA caching allocator behavior, where memory is reserved in blocks/chunks rather than exact tensor-size increments.

### 5. Final allocated memory stayed almost constant

Final allocated memory stayed around 329 MB for all batch sizes.

This means the persistent model memory after generation was almost unchanged.

The additional batch-dependent memory mainly appeared in peak memory during computation.

This again shows why peak memory is more important than final memory for prediction.

### 6. Runtime decreased in this experiment

Runtime decreased from:

- batch_size = 1 -> 1.243 seconds
- batch_size = 8 -> 0.437 seconds

This may seem surprising, but it can happen because larger batches use the GPU more efficiently.

However, this should not be overgeneralized.

Colab runtime can be noisy, and the first run may include warm-up overhead.

The safer conclusion is:

In this experiment, larger batch sizes improved runtime efficiency, but repeated timing experiments would be needed before making a strong performance claim.

## Main Interpretation

Day 8 showed that total memory increases with batch size, but not linearly.

The model weights are fixed and shared across all batch elements.

Batch-dependent memory such as activations, attention tensors, KV cache, input tensors, and temporary buffers grows with batch size.

This is why total memory increased only moderately, while memory per sample decreased sharply.

This result is important for the final memory estimator because batch size must be included as a key input variable.

## Why This Experiment Matters for the Project

This project is about building xMem-inspired memory prediction modules for LLM workloads.

A useful memory estimator must consider:

- model size
- input sequence length
- generated sequence length
- batch size
- dtype
- cache behavior
- allocator behavior

Day 8 specifically targets batch size.

It shows that a naive estimator cannot simply multiply total memory by batch size.

A better estimator must separate fixed model memory from batch-dependent memory.

## Plots Created

The following plots were created and saved:

- plots/day8_distilgpt2_peak_allocated_vs_batch_size.png
- plots/day8_distilgpt2_peak_reserved_vs_batch_size.png
- plots/day8_distilgpt2_runtime_vs_batch_size.png

## Questions Answered

### Q1. How did peak_allocated_MB change from batch size 1 to the largest successful batch size?

Peak allocated memory increased from 335.36 MB at batch size 1 to 384.00 MB at batch size 8.

The increase was 48.64 MB, or about 14.5%.

### Q2. Did memory scale linearly with batch size or sub-linearly?

Memory scaled sub-linearly.

Batch size increased 8 times, but peak allocated memory increased only by about 14.5%.

This happened because model weights are shared across the batch, while only batch-dependent tensors grow with batch size.

### Q3. What happened to peak_allocated_per_sample_MB as batch size increased?

Peak allocated memory per sample decreased sharply.

It decreased from 335.36 MB/sample at batch size 1 to 48.00 MB/sample at batch size 8.

This shows that larger batches can be more memory-efficient per sample.

### Q4. Did peak_reserved_MB increase smoothly or in jumps?

Peak reserved memory increased in chunks:

- 364 MB
- 372 MB
- 396 MB
- 416 MB

This reflects PyTorch CUDA allocator behavior.

### Q5. Why is batch size important for memory prediction?

Batch size is important because it affects activations, attention tensors, KV-cache tensors, input tensors, and temporary buffers.

Model weights stay fixed, but batch-dependent tensors grow with batch size.

Therefore, any LLM memory estimator must include batch size as a key input.

## Day 8 Conclusion

On Day 8, I studied how batch size affects distilgpt2 inference memory.

I fixed input_tokens = 64, max_new_tokens = 64, dtype = fp32, and use_cache = True, while varying batch_size from 1 to 8.

Peak allocated memory increased from 335.36 MB at batch size 1 to 384.00 MB at batch size 8.

This was an increase of 48.64 MB, or about 14.5%.

The scaling was strongly sub-linear.

Although batch size increased 8 times, peak allocated memory increased only 14.5%.

This happened because model weights are loaded once and shared across all samples, while only activations, attention tensors, KV cache, input tensors, and temporary buffers scale with batch size.

Peak allocated memory per sample decreased sharply from 335.36 MB/sample at batch size 1 to 48.00 MB/sample at batch size 8.

Peak reserved memory increased from 364 MB to 416 MB in chunks, showing PyTorch CUDA allocator behavior.

Runtime decreased in this experiment as batch size increased, likely because larger batches improved GPU utilization and the first run may have included warm-up overhead.

This result should not be overgeneralized without repeated timing experiments.

This experiment is important because batch size is one of the key hyperparameters that must be included in the final memory prediction estimator.

The next step is to study precision effects by comparing fp32 and fp16 memory usage.

---


# Day 9 - distilgpt2 Precision Experiment: fp32 vs fp16

## Goal

Study how inference memory changes when model precision changes from fp32 to fp16.

This experiment directly targets the precision-aware / quantization-related part of the project.

fp16 is not full quantization, but it is the correct first step toward building a PrecisionAwareEstimator.

## Fixed Variables

- model_name = distilgpt2
- batch_size = 1
- input_tokens = 128
- max_new_tokens = 128
- use_cache = True
- platform = Google Colab
- GPU = Tesla T4

## Changed Variable

dtype = fp32, fp16

Each dtype was repeated 3 times because runtime can be noisy.

## Why Precision Was Changed

Memory used by model parameters and many tensors depends on dtype.

Basic memory per value:

- fp32 = 4 bytes
- fp16 = 2 bytes

So fp16 is expected to reduce memory significantly.

However, total peak GPU memory may not reduce by exactly 50% because total memory includes more than just model weights.

It also includes:

- activations
- KV cache
- temporary buffers
- input tensors
- CUDA/PyTorch overhead
- allocator reserved memory
- framework-level memory behavior

## Experiment Results

| model_name | dtype | repeat_id | batch_size | input_tokens | max_new_tokens | output_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| distilgpt2 | fp32 | 1 | 1 | 128 | 128 | 256 | 343.74 | 372.00 | 328.99 | 372.00 | 1.487 | False |
| distilgpt2 | fp32 | 2 | 1 | 128 | 128 | 256 | 343.24 | 374.00 | 328.99 | 374.00 | 0.703 | False |
| distilgpt2 | fp32 | 3 | 1 | 128 | 128 | 256 | 343.24 | 374.00 | 328.99 | 374.00 | 0.730 | False |
| distilgpt2 | fp16 | 1 | 1 | 128 | 128 | 256 | 181.25 | 190.00 | 174.87 | 190.00 | 0.991 | False |
| distilgpt2 | fp16 | 2 | 1 | 128 | 128 | 256 | 181.25 | 190.00 | 174.87 | 190.00 | 0.960 | False |
| distilgpt2 | fp16 | 3 | 1 | 128 | 128 | 256 | 181.25 | 190.00 | 174.87 | 190.00 | 0.681 | False |

## Summary Table

| dtype | avg_peak_allocated_MB | avg_peak_reserved_MB | avg_final_allocated_MB | avg_final_reserved_MB | avg_runtime_sec | min_runtime_sec | max_runtime_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 181.25 | 190.00 | 174.87 | 190.00 | 0.877 | 0.681 | 0.991 |
| fp32 | 343.41 | 373.33 | 328.99 | 373.33 | 0.973 | 0.703 | 1.487 |

## Memory Reduction Calculation

Average peak allocated memory:

- fp32 = 343.4067 MB
- fp16 = 181.25 MB

Reduction:

343.4067 - 181.25 = 162.1567 MB

Reduction percentage:

162.1567 / 343.4067 * 100 = 47.22%

So fp16 reduced peak allocated memory by approximately 47.22%.

## Reserved Memory Reduction

Average peak reserved memory:

- fp32 = 373.33 MB
- fp16 = 190.00 MB

Reduction:

373.33 - 190.00 = 183.33 MB

Reduction percentage:

183.33 / 373.33 * 100 = approximately 49.11%

So fp16 reduced peak reserved memory by approximately 49.11%.

## Runtime Result

Average runtime:

- fp32 = 0.973 seconds
- fp16 = 0.877 seconds

Difference:

0.973 - 0.877 = 0.096 seconds

Percentage speed improvement:

0.096 / 0.973 * 100 = approximately 9.86%

So fp16 was slightly faster in this experiment.

However, runtime on Colab is noisy, so this should not be overclaimed without more repeated timing experiments.

## Observations

### 1. fp16 significantly reduced peak allocated memory

Peak allocated memory reduced from 343.41 MB in fp32 to 181.25 MB in fp16.

This is a reduction of 162.16 MB, or approximately 47.22%.

This is close to the expected 50% reduction because fp16 uses 2 bytes per value instead of 4 bytes.

### 2. fp16 also reduced peak reserved memory

Peak reserved memory reduced from 373.33 MB in fp32 to 190.00 MB in fp16.

This is a reduction of approximately 49.11%.

This shows that precision affects not only active tensor memory but also allocator-reserved memory.

### 3. fp16 did not reduce memory by exactly 50%

The reduction was close to 50%, but not exactly 50%.

This is expected because total peak memory includes more than model weights.

Some parts of memory do not scale perfectly with dtype, including:

- CUDA/PyTorch overhead
- allocator behavior
- input token tensors
- temporary buffers
- some framework-level memory
- operations that may not perfectly halve memory

### 4. Final allocated memory also reduced strongly

Final allocated memory reduced from 328.99 MB in fp32 to 174.87 MB in fp16.

This shows that the persistent model-related memory is strongly affected by dtype.

### 5. Runtime was slightly lower in fp16

Average runtime reduced from 0.973 seconds in fp32 to 0.877 seconds in fp16.

This suggests fp16 may be slightly faster in this setup.

However, because Colab timing is noisy, this should be treated as a weak performance observation, not a final conclusion.

## Main Interpretation

Day 9 clearly shows that dtype is a major factor in LLM memory usage.

fp16 significantly reduced both peak allocated memory and peak reserved memory.

This result directly supports the need for a PrecisionAwareEstimator.

A naive estimator may assume fp16 simply halves memory, but the actual reduction was 47.22% for peak allocated memory and 49.11% for peak reserved memory.

So the estimator should account for dtype while also considering allocator behavior and non-parameter memory.

## Why This Experiment Matters for the Project

The project goal is to adapt xMem-style memory prediction for LLM workloads.

One important optimization technique mentioned in the project is quantization/precision optimization.

Day 9 provides experimental evidence that precision changes memory behavior significantly.

This means the final memory prediction system should include dtype as an input variable.

A PrecisionAwareEstimator should adjust:

- model weight memory
- activation memory
- KV-cache memory
- temporary tensor memory
- reserved memory correction

based on dtype.

## Plots Created

The following plots were created and saved:

- plots/day9_distilgpt2_peak_allocated_fp32_vs_fp16.png
- plots/day9_distilgpt2_peak_reserved_fp32_vs_fp16.png
- plots/day9_distilgpt2_runtime_fp32_vs_fp16.png
- plots/day9_runtime_variability_fp32_vs_fp16.png

## Questions Answered

### Q1. What was average peak_allocated_MB for fp32 and fp16?

- fp32 = 343.41 MB
- fp16 = 181.25 MB

### Q2. How much memory reduction did fp16 give?

fp16 reduced peak allocated memory by:

- 162.16 MB
- 47.22%

### Q3. Did peak_reserved_MB reduce as much as peak_allocated_MB?

Peak reserved memory reduced strongly.

- fp32 avg peak reserved = 373.33 MB
- fp16 avg peak reserved = 190.00 MB

Reduction:

- 183.33 MB
- approximately 49.11%

Peak reserved memory reduced slightly more percentage-wise than peak allocated memory.

### Q4. Did fp16 make runtime faster, slower, or noisy?

fp16 was slightly faster in this experiment.

- fp32 avg runtime = 0.973 seconds
- fp16 avg runtime = 0.877 seconds

This is about 9.86% faster.

However, runtime on Colab is noisy, so this should not be treated as a strong performance conclusion.

### Q5. Why does fp16 not always reduce total peak memory by exactly 50%?

Because total peak memory is not only model weights.

It also includes:

- activations
- KV cache
- temporary buffers
- input tensors
- CUDA/PyTorch overhead
- allocator reserved memory
- framework-level memory

Some of these components do not scale perfectly with dtype.

Therefore, fp16 reduces memory strongly, but not always by exactly 50%.

## Day 9 Conclusion

On Day 9, I compared fp32 and fp16 inference memory for distilgpt2 with batch_size = 1, input_tokens = 128, max_new_tokens = 128, and use_cache = True.

The average peak allocated memory decreased from 343.41 MB in fp32 to 181.25 MB in fp16.

This is a reduction of 162.16 MB, or about 47.22%.

The average peak reserved memory decreased from 373.33 MB in fp32 to 190.00 MB in fp16.

This is a reduction of 183.33 MB, or about 49.11%.

This confirms that fp16 significantly reduces memory usage and is close to the expected half-memory behavior, but not exactly 50%.

The reason is that total peak memory includes not only model weights, but also activations, KV cache, temporary buffers, input tensors, CUDA/PyTorch overhead, and allocator effects.

Runtime also decreased slightly from 0.973 seconds in fp32 to 0.877 seconds in fp16, but runtime on Colab is noisy, so this should not be overclaimed.

This experiment directly supports the precision-aware memory estimation part of the project.

The next step is to begin building the first baseline memory estimator using the inference profiling data collected so far.

---


# Day 10 - Clean Inference Dataset and Summary Analysis

## Goal

Clean the inference experiment dataset and prepare it for memory estimator development.

Until Day 9, the project focused on collecting profiling data.

Day 10 focused on organizing that data into a clean format so that prediction modules can be built on top of it.

## Work Done

- Loaded results/inference_runs.csv
- Filtered inference rows
- Removed failed/OOM/error rows
- Converted numeric columns to correct types
- Normalized boolean columns such as oom and use_cache
- Added derived columns:
  - total_tokens
  - generated_tokens
  - allocated_minus_final_MB
  - reserved_minus_allocated_MB
  - peak_allocated_per_sample_MB
  - runtime_per_sample_sec
- Saved clean dataset as results/inference_clean.csv
- Saved model-level summary as results/inference_summary.csv
- Saved experiment counts as results/day10_experiment_counts.csv

## Plots Created

- day10_avg_peak_allocated_by_model.png
- day10_allocated_vs_reserved.png
- day10_distilgpt2_input_token_scaling.png
- day10_dtype_comparison.png

## Key Findings

The cleaned dataset confirms that tiny-gpt2 is useful for pipeline validation but too small for strong memory-scaling conclusions.

distilgpt2 shows clearer memory behavior across input length, generated length, batch size, cache behavior, and dtype.

The fp16 experiment showed a 47.22% reduction in average peak allocated memory compared to fp32.

Batch size scaling was sub-linear because model weights are shared across the batch.

Peak reserved memory often changed in chunks, showing PyTorch allocator behavior.

## Day 10 Conclusion

Day 10 converted raw experiment logs into a clean dataset ready for estimator development.

The next step is to build the first BaseMemoryEstimator and calculate prediction error using MRE.

---


# Day 11 - BaseMemoryEstimator

## Goal

Start the estimator-building phase of the project.

Until Day 10, the project focused on profiling GPU memory and cleaning the inference dataset.

Day 11 started the prediction phase by building a simple BaseMemoryEstimator.

The goal was not to create a perfect estimator immediately, but to create a working baseline that can later be improved using KV-cache, precision-aware, and allocator-correction modules.

## Files Created

- src/estimators/model_config_utils.py
- src/estimators/base_estimator.py
- results/day11_base_estimator_predictions.csv
- plots/day11_base_actual_vs_predicted.png
- plots/day11_base_relative_error_by_model.png

## What model_config_utils.py Does

The file model_config_utils.py extracts model-related information from Hugging Face models and configs.

It includes functions for:

- dtype byte size
- number of layers
- hidden size
- number of attention heads
- vocabulary size
- parameter count
- parameter memory estimate

This is important because the estimator should be model-agnostic.

It should work for tiny-gpt2, distilgpt2, gpt2, OPT-style models, and other Hugging Face causal language models without rewriting the estimator for each model.

## What base_estimator.py Does

The file base_estimator.py contains the BaseMemoryEstimator class.

The BaseMemoryEstimator predicts peak allocated memory using a simple formula based on:

- model parameter memory
- token-dependent memory component
- batch-dependent memory component
- simple overhead factor
- simple constant overhead
- cache multiplier

The estimator uses:

- model_name
- batch_size
- input_tokens
- max_new_tokens
- dtype
- use_cache

as inputs.

It outputs:

- predicted_peak_allocated_MB
- parameter_memory_MB
- token_memory_component_MB
- batch_memory_component_MB
- total_tokens

## Why a Baseline Estimator Is Needed

A baseline estimator is needed because the project must compare simple prediction against improved prediction.

Even if the first estimator is inaccurate, it gives a reference point.

Later modules should improve it:

- KVCacheEstimator
- PrecisionAwareEstimator
- AllocatorCorrectionEstimator
- OptimizerStateEstimator

Without a baseline, there is no clear way to prove that the later modules improve prediction accuracy.

## Dataset Used

The estimator was run on:

results/inference_clean.csv

This cleaned dataset contains successful inference profiling rows from earlier experiments.

It includes experiments across:

- tiny-gpt2
- distilgpt2
- input token variation
- generated token variation
- use_cache True/False
- batch size variation
- fp32/fp16 precision variation

## Prediction Output

Predictions were saved to:

results/day11_base_estimator_predictions.csv

For each row, the output included:

- actual_peak_allocated_MB
- predicted_peak_allocated_MB
- absolute_error_MB
- relative_error_percent
- parameter_memory_MB
- token_memory_component_MB
- batch_memory_component_MB

## Base Estimator Metrics

The BaseMemoryEstimator produced:

- MRE = 6.82%
- Mean relative error = 42.46%
- Max relative error = 95.14%

Exact values:

- Base Estimator MRE: 6.817565007100576
- Base Estimator Mean Relative Error: 42.4640300566115
- Base Estimator Max Relative Error: 95.13842044238285

## Sample Error Pattern

For tiny-gpt2 rows, the estimator predicted around 24.39 MB, while the actual peak allocated memory was around 12.50 MB.

Example:

- actual_peak_allocated_MB = 12.50 MB
- predicted_peak_allocated_MB = 24.39 MB
- relative_error_percent ≈ 95%

This means the estimator badly overpredicted tiny-gpt2 memory.

## Why tiny-gpt2 Error Was High

The estimator used a fixed overhead constant.

For larger models like distilgpt2, this fixed overhead is small compared to the total memory.

But for tiny-gpt2, actual memory is only around 12.5 MB.

So adding a fixed overhead of around 20 MB creates a very large relative error.

This shows that fixed overhead terms work poorly for extremely small models.

## Interpretation of MRE vs Mean Error

The MRE was low:

- MRE = 6.82%

But the mean relative error was much higher:

- Mean relative error = 42.46%

This means the estimator performs reasonably for the median case, but some rows have very high error.

The maximum error was:

- Max relative error = 95.14%

This high max error mostly came from tiny-gpt2 rows.

This shows why looking only at MRE is not enough. Mean error, max error, and per-model error breakdown are also important.

## Main Interpretation

The BaseMemoryEstimator is working, but it is not uniformly accurate.

It performs better on larger model rows and worse on tiny-gpt2 rows.

This is expected because the estimator is intentionally simple.

The purpose of Day 11 was to create a baseline, not a final estimator.

The high error on tiny-gpt2 motivates better correction methods, such as:

- model-size-aware overhead correction
- KV-cache estimation
- precision-aware memory scaling
- allocator correction

## Questions Answered

### Q1. What is the difference between measuring memory and estimating memory?

Measuring memory means actually running the model and recording GPU memory usage.

Estimating memory means predicting memory before or without running the full workload, using model configuration, dtype, batch size, sequence length, and formulas.

### Q2. Why do we need a baseline estimator even if it is inaccurate?

A baseline estimator gives a reference point.

Later, when improved modules are added, their performance can be compared against the baseline.

Without a baseline, there is no way to prove improvement.

### Q3. What is parameter memory?

Parameter memory is the memory occupied by model weights.

Formula:

parameter_memory = number_of_parameters * bytes_per_parameter

For example:

- fp32 uses 4 bytes per parameter
- fp16 uses 2 bytes per parameter

### Q4. Why should the estimator be model-agnostic instead of only for distilgpt2?

The project is about LLM memory prediction generally, not only distilgpt2.

The estimator should work for multiple models such as tiny-gpt2, distilgpt2, gpt2, OPT-125M, and other Hugging Face causal language models.

A model-agnostic estimator reads configuration values automatically instead of hardcoding one model.

### Q5. What was the Base Estimator MRE?

The Base Estimator MRE was:

6.82%

The mean relative error was:

42.46%

The maximum relative error was:

95.14%

## Day 11 Conclusion

On Day 11, I built the first memory prediction module: BaseMemoryEstimator.

This moved the project from only measuring memory to predicting memory.

The estimator produced an MRE of 6.82%, but the mean and maximum errors were high because tiny-gpt2 rows were badly overpredicted.

This is acceptable for a baseline.

The result shows that a simple estimator is not enough and motivates improved modules such as KV-cache estimation, precision-aware correction, and allocator correction.

The next step is to perform detailed error analysis by model and experiment type before improving the estimator.

---


# Day 12 - Base Estimator Error Analysis

## Goal

Analyze where the BaseMemoryEstimator works and where it fails.

Day 11 created the first baseline memory estimator.

Day 12 focused on detailed error breakdown before improving the estimator.

The purpose was not to tune the estimator yet, but to understand the error pattern clearly.

## Files Created

- results/day12_base_error_analysis.csv
- results/day12_error_by_model.csv
- results/day12_error_by_dtype.csv
- results/day12_error_by_cache.csv
- results/day12_error_by_experiment_type.csv
- report/day12_base_error_analysis.md
- plots/day12_mean_error_by_model.png
- plots/day12_error_distribution_by_model.png
- plots/day12_mean_error_by_experiment_type.png
- plots/day12_actual_vs_predicted_by_model.png

## Global Metrics

The BaseMemoryEstimator produced the following global metrics:

- Global MRE: 6.817565007100576%
- Global mean relative error: 42.4640300566115%
- Global max relative error: 95.13842044238284%
- Global min relative error: 1.0521897941643268%

Rounded values:

- Global MRE: 6.82%
- Global mean relative error: 42.46%
- Global max relative error: 95.14%
- Global min relative error: 1.05%

## Error by Model

| model_name | num_rows | mean_error_percent | median_error_percent | max_error_percent | min_error_percent | avg_actual_MB | avg_predicted_MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 28 | 5.282823 | 5.321602 | 8.151976 | 1.052190 | 325.885 | 340.388518 |
| sshleifer/tiny-gpt2 | 20 | 94.517720 | 95.138388 | 95.138420 | 91.312177 | 12.503 | 24.320560 |

## Main Finding

The BaseMemoryEstimator does not fail equally across all models.

It performs reasonably well on distilgpt2 but fails badly on sshleifer/tiny-gpt2.

For distilgpt2:

- Mean relative error: 5.28%
- Median relative error: 5.32%
- Max relative error: 8.15%
- Average actual peak allocated memory: 325.89 MB
- Average predicted peak allocated memory: 340.39 MB

This is a decent result for a simple baseline estimator.

For sshleifer/tiny-gpt2:

- Mean relative error: 94.52%
- Median relative error: 95.14%
- Max relative error: 95.14%
- Average actual peak allocated memory: 12.50 MB
- Average predicted peak allocated memory: 24.32 MB

This is a very poor result.

## Why tiny-gpt2 Error Was So High

The main reason is the fixed overhead term in the BaseMemoryEstimator.

For distilgpt2, actual peak memory is around 300+ MB.

So a fixed overhead of around 20 MB is relatively small compared to the total memory.

But for tiny-gpt2, actual peak memory is only around 12.5 MB.

So the fixed overhead is larger than the actual memory itself.

This causes a very large relative error.

In simple terms:

- Fixed overhead is acceptable for larger models.
- Fixed overhead breaks badly for tiny toy models.

This shows that the estimator needs model-size-aware overhead correction.

## Why MRE and Mean Error Are Very Different

The global MRE was low:

- MRE = 6.82%

But the mean relative error was high:

- Mean relative error = 42.46%

This happened because MRE is the median error.

Median error is less affected by extreme bad rows.

Mean error averages all rows, so the terrible tiny-gpt2 errors increase the mean strongly.

This shows that reporting only MRE is not enough.

A proper evaluation should include:

- MRE
- mean relative error
- max relative error
- per-model error breakdown
- per-experiment-type error breakdown

## Worst Error Pattern

The worst rows came from sshleifer/tiny-gpt2.

For tiny-gpt2 rows:

- actual_peak_allocated_MB ≈ 12.50 MB
- predicted_peak_allocated_MB ≈ 24.32 MB
- relative_error_percent ≈ 95%

The model was consistently overpredicted.

This was not random noise.

It was a systematic error caused by the estimator design.

## Interpretation

The BaseMemoryEstimator is not useless.

It works reasonably for distilgpt2, where the model size is large enough for the formula to make sense.

However, it fails on extremely small models because the fixed overhead dominates prediction.

This is an important research observation.

It shows that toy models and realistic models should not always be treated the same way in memory prediction.

## Estimator Weaknesses Identified

The main weaknesses of the BaseMemoryEstimator are:

1. Fixed overhead is too large for tiny models.
2. The estimator does not explicitly model KV cache.
3. The estimator does not explicitly model allocator behavior.
4. The estimator does not model use_cache behavior accurately.
5. Precision effects are handled only through parameter memory, not through all memory components.

## What This Means for the Project

This analysis gives a clear direction for improvement.

The next estimator should not blindly add a fixed overhead to all models.

Instead, the project should introduce:

- model-size-aware overhead correction
- KV-cache-specific memory estimation
- precision-aware correction
- allocator correction

This moves the project closer to xMem-style memory prediction because it starts separating memory into meaningful components rather than using one rough formula.

## Questions Answered

### Q1. Which model had worse prediction error: tiny-gpt2 or distilgpt2?

sshileifer/tiny-gpt2 had much worse prediction error.

tiny-gpt2:

- Mean error: 94.52%
- Median error: 95.14%

distilgpt2:

- Mean error: 5.28%
- Median error: 5.32%

So the estimator performed much better on distilgpt2.

### Q2. Why does fixed overhead create large relative error for tiny-gpt2?

Because tiny-gpt2 uses only around 12.5 MB of peak allocated memory.

The estimator added a fixed overhead term of around 20 MB.

For such a small model, this overhead is larger than the actual memory itself.

So the prediction almost doubled the actual memory, creating around 95% relative error.

### Q3. Why is MRE lower than mean relative error?

MRE is the median relative error.

It is less affected by extreme bad rows.

Mean relative error averages all rows.

The tiny-gpt2 rows had very high errors around 95%, so they pushed the mean error upward.

That is why:

- MRE = 6.82%
- Mean error = 42.46%

### Q4. What are the top 2 weaknesses of the BaseMemoryEstimator?

The top two weaknesses are:

1. Fixed overhead is too large for tiny models.
2. It does not explicitly model important memory components like KV cache, allocator behavior, and use_cache effects.

### Q5. What module should be built next to improve the estimator?

The next improvement should be model-size-aware overhead correction or an improved base estimator.

After that, the next modules should be:

- KVCacheEstimator
- PrecisionAwareEstimator
- AllocatorCorrectionEstimator

## Day 12 Conclusion

On Day 12, I analyzed the BaseMemoryEstimator error pattern in detail.

The global MRE was 6.82%, but the mean relative error was much higher at 42.46%.

This happened because the estimator performed well on distilgpt2 but failed badly on tiny-gpt2.

For distilgpt2, the estimator achieved a mean error of 5.28% and median error of 5.32%, which is reasonable for a simple baseline.

For tiny-gpt2, the estimator had a mean error of 94.52% and median error of 95.14%.

The main cause was the fixed overhead term, which is too large relative to tiny-gpt2's actual memory.

The key conclusion is that the BaseMemoryEstimator is usable for larger models like distilgpt2, but fails on extremely small models.

This motivates model-size-aware overhead correction and separate evaluation for toy models versus realistic models.

The next step is to improve the base estimator before adding KV-cache and allocator-correction modules.

---


# Day 13 - Improved BaseMemoryEstimator

## Goal

Improve the baseline estimator by fixing the biggest weakness found on Day 12.

Day 12 showed that the fixed overhead term caused very high error for tiny-gpt2.

Day 13 replaced the fixed overhead with model-size-aware overhead.

The goal was to improve the BaseMemoryEstimator before adding more advanced modules like KV-cache estimation, precision-aware correction, and allocator correction.

## Files Created

- src/estimators/improved_base_estimator.py
- results/day13_improved_base_predictions.csv
- results/day13_base_vs_improved_comparison.csv
- results/day13_model_error_comparison.csv
- report/day13_improved_base_estimator.md
- plots/day13_mean_error_base_vs_improved.png
- plots/day13_model_error_base_vs_improved.png
- plots/day13_improved_actual_vs_predicted.png

## Main Change

The Day 11 BaseMemoryEstimator used a fixed overhead constant.

That fixed overhead worked poorly for very small models like tiny-gpt2.

The Day 13 ImprovedBaseMemoryEstimator uses model-size-aware overhead.

Instead of using the same overhead for every model, the improved estimator uses:

overhead = parameter_memory_MB * overhead_ratio

with minimum and maximum overhead limits.

This makes the overhead depend on model size.

## Why This Change Was Needed

In Day 12, tiny-gpt2 had very high prediction error.

Reason:

- tiny-gpt2 actual peak memory was around 12.5 MB
- Day 11 estimator added a fixed overhead around 20 MB
- this overhead was larger than the actual model memory itself
- therefore, tiny-gpt2 was badly overpredicted

For larger models like distilgpt2, the same overhead was not too harmful because total memory was around 300+ MB.

So the fixed overhead problem mainly affected tiny toy models.

## Global Metric Comparison

| estimator | MRE | Mean Error | Max Error | Min Error |
|---|---:|---:|---:|---:|
| Day 11 Base | 6.817565 | 42.464030 | 95.138420 | 1.052190 |
| Day 13 Improved Base | 5.673501 | 30.371345 | 69.797927 | 0.380620 |

## Global Improvement

The improved estimator performed better overall.

Changes:

- MRE improved from 6.82% to 5.67%
- Mean error improved from 42.46% to 30.37%
- Max error improved from 95.14% to 69.80%

This shows that model-size-aware overhead helped reduce the overall prediction error.

## Model-wise Error Comparison

| model_name | base_mean_error | base_median_error | base_max_error | improved_mean_error | improved_median_error | improved_max_error |
|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 5.282823 | 5.321602 | 8.151976 | 2.570964 | 1.350309 | 11.485668 |
| sshleifer/tiny-gpt2 | 94.517720 | 95.138388 | 95.138420 | 69.291879 | 69.193916 | 69.797927 |

## distilgpt2 Result

For distilgpt2, the improved estimator worked very well.

Mean error reduced from:

5.28% to 2.57%

Median error reduced from:

5.32% to 1.35%

This is a strong improvement for a simple base estimator.

It shows that model-size-aware overhead improves prediction for the more realistic model.

## tiny-gpt2 Result

For tiny-gpt2, the error also reduced, but it is still high.

Mean error reduced from:

94.52% to 69.29%

Median error reduced from:

95.14% to 69.19%

So the improved estimator helped, but tiny-gpt2 is still poorly predicted.

## Important Observation About tiny-gpt2

The error pattern changed.

In Day 11, the estimator overpredicted tiny-gpt2.

Example:

- actual memory was around 12.50 MB
- predicted memory was around 24.39 MB

In Day 13, the estimator underpredicted tiny-gpt2.

Example:

- actual memory was around 12.50 MB
- predicted memory was around 3.85 MB

So the fixed overhead problem was reduced, but now the minimum overhead is too low for tiny-gpt2.

This means tiny toy models may need separate handling or better allocator/overhead correction.

## Sample tiny-gpt2 Improved Rows

| model_name | actual_peak_allocated_MB | predicted_peak_allocated_MB | parameter_memory_MB | model_size_overhead_MB | absolute_error_MB | relative_error_percent |
|---|---:|---:|---:|---:|---:|---:|
| sshleifer/tiny-gpt2 | 12.50 | 3.850760 | 0.775253 | 1.0 | 8.649240 | 69.193917 |
| sshleifer/tiny-gpt2 | 12.50 | 3.850761 | 0.775253 | 1.0 | 8.649239 | 69.193909 |
| sshleifer/tiny-gpt2 | 12.50 | 3.850760 | 0.775253 | 1.0 | 8.649240 | 69.193921 |
| sshleifer/tiny-gpt2 | 12.50 | 3.850760 | 0.775253 | 1.0 | 8.649240 | 69.193917 |
| sshleifer/tiny-gpt2 | 12.50 | 3.850761 | 0.775253 | 1.0 | 8.649239 | 69.193909 |

## Main Interpretation

The Day 13 ImprovedBaseMemoryEstimator improved the global metrics compared to the Day 11 baseline.

It especially improved distilgpt2 prediction.

For distilgpt2, mean error became only 2.57%, which is strong for a simple base estimator.

For tiny-gpt2, the estimator improved compared to Day 11, but still has high error.

The reason is that tiny-gpt2 is an extremely small toy model, and small overhead changes create very large relative error.

This shows that tiny-gpt2 is useful for testing the pipeline, but not ideal for judging realistic estimator quality.

## What This Means for the Project

This is the first clear estimator improvement result.

The project now has:

- a baseline estimator
- an improved base estimator
- comparison between both
- per-model error analysis
- evidence that model-size-aware overhead improves prediction

This moves the project closer to xMem-style memory prediction because the estimator now treats overhead more carefully instead of using a fixed constant for every model.

## Remaining Weaknesses

The improved estimator is still not final.

Remaining issues:

1. tiny-gpt2 is still badly underpredicted.
2. KV-cache memory is not explicitly modeled.
3. Allocator behavior is not explicitly modeled.
4. use_cache behavior is still simplified.
5. precision effects need a dedicated module.

## Questions Answered

### Q1. Why did the fixed overhead hurt tiny-gpt2 predictions?

Because tiny-gpt2 uses only around 12.5 MB peak allocated memory.

A fixed overhead of around 20 MB is larger than the actual memory itself.

So the Day 11 estimator overpredicted tiny-gpt2 badly.

### Q2. What is model-size-aware overhead?

Model-size-aware overhead means the overhead depends on model size instead of using the same constant for every model.

Instead of:

overhead = 20 MB for every model

the improved estimator uses:

overhead = parameter_memory_MB * overhead_ratio

with minimum and maximum overhead limits.

### Q3. Did the improved estimator reduce tiny-gpt2 error?

Yes.

tiny-gpt2 mean error reduced from 94.52% to 69.29%.

tiny-gpt2 median error reduced from 95.14% to 69.19%.

However, the error is still high because the estimator now underpredicts tiny-gpt2.

### Q4. Did the improved estimator hurt distilgpt2 prediction or improve it?

It improved distilgpt2 prediction clearly.

distilgpt2 mean error reduced from 5.28% to 2.57%.

distilgpt2 median error reduced from 5.32% to 1.35%.

### Q5. What should the next estimator module be?

The next estimator module should be the KVCacheEstimator.

This will explicitly model KV-cache memory during generation.

After that, precision-aware and allocator-correction modules should be added.

## Day 13 Conclusion

On Day 13, I improved the BaseMemoryEstimator by replacing fixed overhead with model-size-aware overhead.

The improved estimator reduced global MRE from 6.82% to 5.67%.

It reduced mean error from 42.46% to 30.37%.

It reduced max error from 95.14% to 69.80%.

For distilgpt2, the estimator improved strongly, reducing mean error from 5.28% to 2.57%.

For tiny-gpt2, error also improved, reducing from 94.52% to 69.29%, but it is still high.

The improved estimator now underpredicts tiny-gpt2, showing that tiny toy models still need special handling or better overhead/allocator correction.

Overall, Day 13 produced the first clear estimator improvement result.

The next step is to build the KVCacheEstimator.

---


# Day 14 - KVCacheEstimator

## Goal

Build a KV-cache-specific estimator and test whether adding KV-cache memory improves prediction.

Day 14 is important because KV cache is one of the main LLM-specific memory components.

Until now, the estimator used general components like parameter memory, token-related memory, batch-related memory, and overhead.

Day 14 added an explicit KV-cache memory component.

## Files Created

- src/estimators/kv_cache_estimator.py
- results/day14_kv_cache_estimates.csv
- results/day14_improved_plus_kv_predictions.csv
- results/day14_improved_vs_kv_comparison.csv
- results/day14_model_comparison.csv
- results/day14_error_by_experiment_type.csv
- report/day14_kv_cache_estimator.md
- plots/day14_mean_error_improved_vs_kv.png
- plots/day14_model_error_improved_vs_kv.png
- plots/day14_distilgpt2_kv_cache_vs_total_tokens.png
- plots/day14_actual_vs_predicted_kv.png

## What KV Cache Is

During autoregressive generation, the model generates tokens one by one.

For each generated token, the model uses attention over previous tokens.

To avoid recomputing attention information for all previous tokens every time, the model can store previous key and value tensors.

This stored memory is called KV cache.

KV cache helps generation speed, but it also uses memory.

## KV Cache Formula Used

KV cache memory was estimated using:

batch_size * total_tokens * num_layers * hidden_size * 2 * bytes_per_element

where:

- total_tokens = input_tokens + max_new_tokens
- num_layers = number of Transformer layers
- hidden_size = model hidden dimension
- 2 = key and value tensors
- bytes_per_element depends on dtype
  - fp32 = 4 bytes
  - fp16 = 2 bytes

## Why kv_weight Was Used

The ImprovedBaseMemoryEstimator already had a rough token-memory component.

If full KV-cache memory was added directly, some token-related memory could be double-counted.

To avoid aggressive double counting, a partial KV contribution was used:

predicted_memory = improved_base_prediction + kv_weight * estimated_kv_cache_MB

The value used was:

kv_weight = 0.5

This means only half of the estimated KV-cache memory was added to the improved base prediction.

## Global Comparison

| estimator | MRE | Mean Error | Max Error | Min Error |
|---|---:|---:|---:|---:|
| Day 13 Improved Base | 5.673501 | 30.371345 | 69.797927 | 0.380620 |
| Day 14 Improved + KV | 3.752376 | 30.187126 | 69.797927 | 0.150284 |

## Global Result

Adding the KV-cache component improved the global MRE.

MRE improved from:

5.67% to 3.75%

Mean error improved slightly from:

30.37% to 30.19%

Minimum error improved from:

0.38% to 0.15%

Max error stayed the same at:

69.80%

This means the KV module helped the median prediction, but did not fix the worst-case error.

The worst-case error still comes from tiny-gpt2, where KV-cache memory is extremely small and the main issue is overhead modeling.

## Model-wise Comparison

| model_name | day13_mean_error | day13_median_error | day13_max_error | day14_mean_error | day14_median_error | day14_max_error |
|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 2.570964 | 1.350309 | 11.485668 | 2.267462 | 2.016841 | 8.995498 |
| sshleifer/tiny-gpt2 | 69.291879 | 69.193916 | 69.797927 | 69.274654 | 69.183753 | 69.797927 |

## distilgpt2 Result

For distilgpt2, adding KV-cache estimation helped.

Mean error improved from:

2.57% to 2.27%

Max error improved from:

11.49% to 9.00%

Median error became slightly worse:

1.35% to 2.02%

This is acceptable because the overall distilgpt2 prediction is still strong, and the max error reduced.

## tiny-gpt2 Result

For tiny-gpt2, the KV module made almost no difference.

Mean error changed only slightly:

69.29% to 69.27%

Median error changed only slightly:

69.19% to 69.18%

Max error stayed the same:

69.80%

This happened because tiny-gpt2 has extremely small KV-cache memory.

Its estimated KV-cache value is around only 0.005 MB in many rows.

So KVCacheEstimator cannot fix tiny-gpt2 errors.

The main issue for tiny-gpt2 is still overhead and allocator modeling.

## Experiment-type Error Breakdown

| model_name | experiment_type | mean_error | median_error | max_error | avg_kv_cache_MB |
|---|---|---:|---:|---:|---:|
| distilgpt2 | batch_size_variation | 1.384443 | 0.930416 | 3.072630 | 21.000000 |
| distilgpt2 | cache_comparison | 2.422627 | 2.422627 | 2.422627 | 0.000000 |
| distilgpt2 | generated_token_variation | 1.869845 | 2.016841 | 2.021401 | 5.843750 |
| distilgpt2 | input_token_variation | 3.011497 | 2.658702 | 8.995498 | 6.508929 |
| distilgpt2 | other | 0.287431 | 0.336080 | 0.336080 | 9.000000 |
| distilgpt2 | precision_fp16 | 4.432121 | 4.432121 | 4.432121 | 4.500000 |
| sshleifer/tiny-gpt2 | cache_comparison | 69.797927 | 69.797927 | 69.797927 | 0.000000 |
| sshleifer/tiny-gpt2 | generated_token_variation | 69.176099 | 69.174362 | 69.185126 | 0.005219 |
| sshleifer/tiny-gpt2 | input_token_variation | 69.187835 | 69.186105 | 69.201308 | 0.004937 |

## Main Experiment-type Observation

For distilgpt2, generated-token variation rows had low error after adding KV cache:

Mean error = 1.87%

This is important because generated-token variation is directly related to KV-cache memory.

The batch-size variation rows also had low error:

Mean error = 1.38%

This suggests that the estimator is performing reasonably well on the more meaningful distilgpt2 experiments.

For tiny-gpt2, the error stayed high across all experiment types because KV-cache memory is too small to matter.

## Main Interpretation

The KVCacheEstimator improved the prediction quality, especially for distilgpt2.

The global MRE improved from 5.67% to 3.75%.

For distilgpt2, mean error improved from 2.57% to 2.27%.

This shows that adding an explicit KV-cache component helps LLM-relevant rows.

However, KV-cache modeling does not solve all problems.

It does not fix tiny-gpt2 because tiny-gpt2 has almost no KV-cache memory compared to its fixed overhead and allocator effects.

This means the project needs both:

- KV-cache modeling for realistic LLMs
- allocator/overhead correction for small-model and reserved-memory behavior

## Why This Matters for the Project

The original project is about adapting xMem-style memory prediction to large model workloads.

KV cache is one of the main memory components that is specific to autoregressive LLM inference.

Day 14 adds the first LLM-specific estimator module.

This moves the project beyond simple parameter-memory estimation.

The estimator now includes:

- improved base memory
- model-size-aware overhead
- explicit KV-cache memory

This is a meaningful step toward LLM-specific memory prediction.

## Questions Answered

### Q1. What is KV cache?

KV cache is the stored key and value tensors from previous tokens during autoregressive generation.

It allows the model to reuse previous attention information instead of recomputing it again and again.

### Q2. Why does KV-cache memory depend on total tokens?

KV cache stores key/value tensors for tokens in the sequence.

More input tokens and more generated tokens mean more stored key/value tensors.

So KV-cache memory grows with total tokens.

The rough formula is:

batch_size * total_tokens * num_layers * hidden_size * 2 * bytes_per_element

### Q3. Why did we use kv_weight = 0.5 instead of adding full KV cache?

Because the improved base estimator already includes a rough token-memory component.

Adding full KV-cache memory directly could double-count token-related memory.

So kv_weight = 0.5 was used as a controlled partial contribution.

### Q4. Did adding KV cache improve global error or make it worse?

It improved global error.

MRE improved from 5.67% to 3.75%.

Mean error improved slightly from 30.37% to 30.19%.

So the KV module helped overall.

### Q5. Which rows should benefit most from KV-cache modeling?

Rows with:

- larger generated tokens
- larger total tokens
- larger batch size
- larger hidden size
- more layers
- use_cache=True

In the current data, distilgpt2 generated-token variation and input-token variation rows are the most relevant for KV-cache modeling.

## Day 14 Conclusion

On Day 14, I built the KVCacheEstimator.

The KV-cache estimator uses model configuration values like number of layers and hidden size, along with batch size, total tokens, dtype, and use_cache setting.

It was combined with the Day 13 ImprovedBaseMemoryEstimator using kv_weight = 0.5.

The global MRE improved from 5.67% to 3.75%.

For distilgpt2, mean error improved from 2.57% to 2.27%, and max error improved from 11.49% to 9.00%.

For tiny-gpt2, the improvement was negligible because its KV-cache memory is extremely small.

The key conclusion is that KV-cache modeling improves LLM-relevant rows, especially distilgpt2, but tiny toy models still need separate overhead or allocator correction.

This is the first clear LLM-specific estimator module in the project.

The next step is to build the PrecisionAwareEstimator using the fp32 vs fp16 results.

---


# Day 15 - PrecisionAwareEstimator

## Goal

Build a precision-aware estimator for fp32 vs fp16 memory prediction.

Day 15 focused on modeling dtype effects, especially how fp16 changes memory compared to fp32.

This connects to the optimization/quantization part of the project. fp16 is not full quantization, but it is the correct first step for precision-aware memory estimation.

## Files Created

- src/estimators/precision_estimator.py
- results/day15_precision_estimates.csv
- results/day15_precision_aware_predictions.csv
- results/day15_precision_comparison.csv
- results/day15_dtype_error_comparison.csv
- report/day15_precision_aware_estimator.md
- plots/day15_mre_day14_vs_precision.png
- plots/day15_mean_error_by_dtype.png
- plots/day15_fp16_actual_vs_predicted.png

## Motivation

Day 9 showed that fp16 significantly reduced distilgpt2 memory.

In Day 9:

- fp32 average peak allocated memory = 343.41 MB
- fp16 average peak allocated memory = 181.25 MB
- reduction = 162.16 MB
- reduction percentage = 47.22%

This showed that dtype has a large effect on memory usage.

The goal of Day 15 was to convert this observation into a reusable estimator module.

## What PrecisionAwareEstimator Does

The PrecisionAwareEstimator models memory changes caused by dtype.

Basic dtype scale:

- fp32 = 1.0
- fp16 = 0.5
- bf16 = 0.5
- int8 = 0.25

However, it does not simply divide total memory by 2 for fp16.

This is because total peak memory includes more than just dtype-scaled tensors.

It includes:

- model weights
- activations
- KV cache
- temporary buffers
- input tensors
- CUDA/PyTorch overhead
- allocator behavior
- framework-level memory

Some of these components scale with dtype, but some do not scale perfectly.

## Method Used

The estimator separates memory into:

- scaling memory
- non-scaling memory

Scaling memory is affected by dtype.

Non-scaling memory represents overhead and memory components that may not reduce perfectly when dtype changes.

For fp16, the estimator applies dtype scaling and an empirical correction.

The correction was used because real fp16 memory reduction was close to, but not exactly, 50%.

## Global Comparison

| estimator | MRE | Mean Error | Max Error | Min Error |
|---|---:|---:|---:|---:|
| Day 14 Improved + KV | 3.752376 | 30.187126 | 69.797927 | 0.150284 |
| Day 15 Precision Aware | 4.204764 | 30.243674 | 69.797927 | 0.150284 |

## Global Result

The global metrics did not improve.

MRE changed from:

3.75% to 4.20%

Mean error changed from:

30.19% to 30.24%

Max error stayed the same:

69.80%

This does not mean the precision module failed.

The reason is that the dataset has only 3 fp16 rows out of 48 total rows.

So global metrics are mostly dominated by fp32 rows and tiny-gpt2 rows.

The better way to judge Day 15 is to look at fp16-specific error.

## Dtype-wise Error Comparison

| dtype | num_rows | mean_error | median_error | max_error | avg_actual_MB | avg_predicted_MB |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 3 | 5.336898 | 5.336898 | 5.336898 | 181.250000 | 171.576872 |
| fp32 | 45 | 31.904126 | 2.871159 | 69.797927 | 196.246444 | 192.681743 |

## fp16 Result

For fp16 rows:

- actual peak allocated memory = 181.25 MB
- predicted peak allocated memory = 171.58 MB
- absolute error = 9.67 MB
- relative error = 5.34%

This is a reasonable result for the first precision-aware estimator.

However, the estimator underpredicted fp16 memory slightly.

## fp16 Row Details

| model_name | batch_size | input_tokens | max_new_tokens | actual_peak_allocated_MB | day14_predicted_MB | fp32_reference_predicted_MB | precision_adjusted_predicted_MB | absolute_error_MB | relative_error_percent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 1 | 128 | 128 | 181.25 | 173.21678 | 344.39356 | 171.576872 | 9.673128 | 5.336898 |
| distilgpt2 | 1 | 128 | 128 | 181.25 | 173.21678 | 344.39356 | 171.576872 | 9.673128 | 5.336898 |
| distilgpt2 | 1 | 128 | 128 | 181.25 | 173.21678 | 344.39356 | 171.576872 | 9.673128 | 5.336898 |

## Important Observation

The Day 15 precision-aware prediction was slightly worse than the Day 14 prediction for fp16.

Day 14 fp16 prediction:

173.22 MB

Day 15 fp16 prediction:

171.58 MB

Actual fp16 memory:

181.25 MB

So the PrecisionAwareEstimator reduced the prediction too much.

This means the fp16 empirical correction was slightly too aggressive.

## Main Interpretation

Day 15 successfully created a dtype-aware estimator module.

The module gives a reasonable fp16 prediction with 5.34% relative error.

However, it did not improve global metrics because fp16 rows are a small part of the dataset.

The module also slightly underpredicted fp16 memory, which means the precision correction needs tuning later.

The main value of Day 15 is that the prediction pipeline now has a separate precision-aware component.

This is important because dtype is a key factor in LLM memory usage.

## Why fp16 Does Not Reduce Memory Exactly by 50%

fp16 uses 2 bytes per value while fp32 uses 4 bytes per value.

So many tensor memories reduce approximately by half.

But total peak memory is not only tensor values.

It also includes:

- CUDA overhead
- PyTorch allocator behavior
- input tensors
- temporary buffers
- framework memory
- metadata
- some operations that may not perfectly halve memory

Therefore, fp16 usually reduces memory strongly, but not exactly by 50%.

In this project, Day 9 showed a 47.22% reduction in peak allocated memory.

## What This Means for the Project

This module supports the precision/optimization part of the project.

The project now has:

- ImprovedBaseMemoryEstimator
- KVCacheEstimator
- PrecisionAwareEstimator

The estimator pipeline is becoming more modular.

Each module handles a different memory factor:

- base estimator handles parameter and rough workload memory
- KVCacheEstimator handles autoregressive cache memory
- PrecisionAwareEstimator handles dtype memory scaling

## Remaining Weaknesses

The PrecisionAwareEstimator is not final.

Remaining issues:

1. fp16 correction is slightly too aggressive.
2. There are only 3 fp16 rows, so the dataset is too small for strong conclusions.
3. Global metrics are not strongly affected because most rows are fp32.
4. Allocator and reserved memory behavior are still not modeled.

## Questions Answered

### Q1. Why does fp16 reduce memory compared to fp32?

fp32 uses 4 bytes per value.

fp16 uses 2 bytes per value.

So model weights, activations, KV cache, and many temporary tensors use less memory in fp16.

### Q2. Why does fp16 not reduce total peak memory by exactly 50%?

Because total peak memory includes more than dtype-scaled tensors.

It also includes CUDA/PyTorch overhead, allocator behavior, input tensors, temporary buffers, metadata, and framework-level memory.

These components may not scale perfectly with dtype.

### Q3. Did the PrecisionAwareEstimator improve fp16 prediction?

It produced a reasonable fp16 prediction, but compared to Day 14 it slightly worsened the fp16 prediction.

Day 14 predicted 173.22 MB.

Day 15 predicted 171.58 MB.

Actual memory was 181.25 MB.

So Day 15 underpredicted fp16 memory slightly more.

### Q4. What was the fp16 prediction error?

For fp16:

- absolute error = 9.67 MB
- relative error = 5.34%

### Q5. What should the next estimator module be?

The next module should be the AllocatorCorrectionEstimator.

This is needed because remaining errors are strongly connected to overhead and allocator behavior, especially peak reserved memory and tiny-gpt2 prediction issues.

## Day 15 Conclusion

On Day 15, I built the PrecisionAwareEstimator to model fp32 vs fp16 memory behavior.

The global MRE changed from 3.75% to 4.20%, so global metrics did not improve.

However, this is because only 3 out of 48 rows are fp16.

The more important result is the fp16-specific prediction.

For fp16 rows, the estimator predicted 171.58 MB while the actual peak allocated memory was 181.25 MB.

This gave a relative error of 5.34%.

The estimator underpredicted fp16 memory slightly, meaning the empirical correction was a bit too aggressive.

Overall, the precision-aware module is useful because it explicitly models dtype effects, which are important for LLM memory prediction and optimization.

The next step is to build an AllocatorCorrectionEstimator to model peak reserved memory and allocator behavior.

---


# Day 16 - AllocatorCorrectionEstimator

## Goal

Build an allocator correction estimator to predict peak reserved memory from predicted peak allocated memory.

Until Day 15, most estimators focused on predicting peak allocated memory.

Day 16 focused on peak reserved memory, which is important because PyTorch CUDA memory behavior depends not only on active tensor memory but also on allocator-reserved memory.

## Files Created

- src/estimators/allocator_correction.py
- results/day16_allocator_correction_predictions.csv
- results/day16_allocator_metrics.csv
- results/day16_allocator_error_by_model.csv
- results/day16_padding_by_model.csv
- results/day16_padding_by_dtype.csv
- report/day16_allocator_correction.md
- plots/day16_actual_allocated_vs_reserved.png
- plots/day16_actual_vs_predicted_reserved.png
- plots/day16_reserved_error_by_model.png
- plots/day16_padding_ratio_by_model.png

## Why Allocator Correction Is Needed

PyTorch reports two important memory values:

1. allocated memory
2. reserved memory

Allocated memory means active tensor memory.

Reserved memory means memory held by PyTorch's CUDA caching allocator.

Reserved memory is usually larger than allocated memory because PyTorch reserves memory blocks and reuses them instead of immediately returning memory to the GPU.

This means reserved memory often increases in jumps or chunks instead of increasing smoothly.

For xMem-style prediction, predicting only allocated tensor memory is incomplete.

A memory estimator should also account for allocator behavior because reserved memory affects real GPU memory pressure and OOM risk.

## Padding by Model

| model_name | num_rows | avg_allocated_MB | avg_reserved_MB | avg_padding_MB | median_padding_MB | avg_padding_ratio | median_padding_ratio | max_padding_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 28 | 325.885 | 354.642857 | 28.757857 | 28.7 | 0.086427 | 0.085401 | 0.124451 |
| sshleifer/tiny-gpt2 | 20 | 12.503 | 26.000000 | 13.497000 | 13.5 | 1.079502 | 1.080000 | 1.080000 |

## Main Padding Observation

distilgpt2 had a median padding ratio of:

0.08540076335877858

This means reserved memory was usually around 8.54% higher than allocated memory.

tiny-gpt2 had a median padding ratio of:

1.08

This means reserved memory was around 108% higher than allocated memory.

This is because tiny-gpt2 is extremely small, but PyTorch still reserves a minimum memory block.

So tiny-gpt2 strongly distorts allocator ratios.

## Padding by dtype

| dtype | num_rows | avg_allocated_MB | avg_reserved_MB | avg_padding_MB | median_padding_MB | avg_padding_ratio | median_padding_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 3 | 181.250000 | 190.000000 | 8.750000 | 8.75 | 0.048276 | 0.048276 |
| fp32 | 45 | 196.246444 | 219.555556 | 23.309111 | 26.78 | 0.530337 | 0.124451 |

## Padding Ratios Calculated

The following padding ratios were calculated:

- Overall median padding ratio: 0.11708204651343407
- Overall mean padding ratio: 0.5002078917658107
- distilgpt2 median padding ratio: 0.08540076335877858
- distilgpt2 mean padding ratio: 0.08642667365129687

The overall mean padding ratio is too high because tiny-gpt2 distorts the average.

Therefore, the allocator correction used the distilgpt2 median padding ratio:

0.08540076335877858

This was the correct choice because distilgpt2 is the more realistic model.

## Allocator Correction Method

The allocator estimator predicts reserved memory using:

predicted_reserved_MB = predicted_allocated_MB + allocator_padding

The allocator padding is calculated using a padding ratio.

For Day 16, the padding ratio used was:

0.08540076335877858

This means around 8.54% extra memory was added on top of predicted allocated memory.

The result was then rounded to simulate allocator chunk behavior.

## Reserved Memory Metrics

The reserved memory prediction gave:

- Reserved memory MRE: 4.545454545454545%
- Reserved memory mean error: 23.81308834025933%
- Reserved memory max error: 53.84615384615385%
- Reserved memory min error: 0.0%

Rounded:

- Reserved memory MRE: 4.55%
- Reserved memory mean error: 23.81%
- Reserved memory max error: 53.85%
- Reserved memory min error: 0.00%

## Reserved Error by Model

| model_name | num_rows | mean_reserved_error | median_reserved_error | max_reserved_error | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 28 | 2.360899 | 2.197802 | 5.729167 | 354.642857 | 354.071429 |
| sshleifer/tiny-gpt2 | 20 | 53.846154 | 53.846154 | 53.846154 | 26.000000 | 12.000000 |

## distilgpt2 Result

For distilgpt2, allocator correction worked well.

Mean reserved-memory error:

2.36%

Median reserved-memory error:

2.20%

Average actual reserved memory:

354.64 MB

Average predicted reserved memory:

354.07 MB

This is a strong result.

It shows that the allocator correction is useful for realistic models like distilgpt2.

## tiny-gpt2 Result

For tiny-gpt2, allocator correction performed poorly.

Mean reserved-memory error:

53.85%

Actual reserved memory:

26 MB

Predicted reserved memory:

12 MB

This happened because tiny-gpt2 is very small, but PyTorch still appears to reserve a minimum memory block.

The estimator underpredicted tiny-gpt2 reserved memory because it did not include a special reserved-memory floor for very small models.

## Main Interpretation

Day 16 showed that allocator behavior is very different for tiny toy models and more realistic models.

For distilgpt2, reserved memory is roughly allocated memory plus 8.54% padding.

For tiny-gpt2, reserved memory is dominated by PyTorch's minimum allocator block behavior.

This explains why tiny-gpt2 repeatedly behaves badly in prediction tasks.

It is useful for testing the pipeline, but not reliable for judging realistic memory-estimator quality.

## Why MRE and Mean Error Differ

The global reserved-memory MRE was low:

4.55%

But the mean error was much higher:

23.81%

This happened because distilgpt2 rows were predicted well, while tiny-gpt2 rows had very high reserved-memory error.

The same pattern appeared in earlier estimator analysis.

This shows again that per-model error breakdown is necessary.

## What This Means for the Project

The project now includes allocator-awareness.

The estimator pipeline has modules for:

- improved base memory prediction
- KV-cache memory estimation
- precision-aware memory adjustment
- allocator correction for reserved memory

This is important because xMem-style prediction should not only estimate tensor memory.

It should also consider allocator behavior and peak memory pressure.

## Remaining Weaknesses

The AllocatorCorrectionEstimator is still not final.

Remaining issues:

1. tiny-gpt2 needs a reserved-memory floor or special handling.
2. allocator behavior is still approximated using a simple padding ratio.
3. PyTorch allocator jumps are not perfectly modeled.
4. reserved memory prediction may change across GPUs and runtimes.
5. more realistic models are needed for stronger validation.

## Questions Answered

### Q1. What is the difference between allocated memory and reserved memory?

Allocated memory is memory currently used by active tensors.

Reserved memory is memory held by PyTorch's CUDA caching allocator.

Reserved memory includes allocated memory plus extra memory blocks kept for reuse.

### Q2. Why is reserved memory usually larger than allocated memory?

Because PyTorch does not immediately return freed GPU memory to the system.

It keeps memory blocks reserved so that future operations can reuse them faster.

This makes reserved memory larger than active allocated memory.

### Q3. What padding ratio did you use?

The allocator correction used the distilgpt2 median padding ratio:

0.08540076335877858

This is approximately 8.54%.

### Q4. What was the reserved memory MRE?

Reserved memory MRE was:

4.545454545454545%

Rounded:

4.55%

### Q5. Why is allocator correction important for xMem-style prediction?

Allocator correction is important because actual GPU memory pressure is not only active tensor memory.

PyTorch reserves memory in chunks, and this affects peak memory and OOM risk.

An xMem-style estimator should account for this allocator behavior instead of predicting only ideal tensor memory.

## Day 16 Conclusion

On Day 16, I built the AllocatorCorrectionEstimator.

The estimator predicts peak reserved memory by adding allocator padding to predicted allocated memory.

The correction used the distilgpt2 median padding ratio of 0.0854 because tiny-gpt2 strongly distorts allocator ratios.

For distilgpt2, allocator prediction worked well.

Mean reserved-memory error was 2.36%, median error was 2.20%, average actual reserved memory was 354.64 MB, and average predicted reserved memory was 354.07 MB.

For tiny-gpt2, allocator prediction was poor.

The mean reserved-memory error was 53.85% because tiny-gpt2 had actual reserved memory of 26 MB while predicted reserved memory was only 12 MB.

This happens because PyTorch appears to reserve a minimum memory block even for very small models.

The global reserved-memory MRE was 4.55%, but mean error was 23.81% due to tiny-gpt2.

This confirms that allocator correction works well for realistic models like distilgpt2, while tiny toy models need a separate reserved-memory floor or special handling.

The next step is to combine ImprovedBaseMemoryEstimator, KVCacheEstimator, PrecisionAwareEstimator, and AllocatorCorrectionEstimator into a single CombinedInferenceEstimator.

---


# Day 17 - CombinedInferenceEstimator

## Goal

Combine the inference estimator modules into one reusable prediction pipeline.

Until Day 16, the project had separate estimator modules:

- ImprovedBaseMemoryEstimator
- KVCacheEstimator
- PrecisionAwareEstimator
- AllocatorCorrectionEstimator

Day 17 integrated these modules into one CombinedInferenceEstimator.

The combined estimator predicts both:

- peak allocated memory
- peak reserved memory

This is important because allocated memory captures active tensor memory, while reserved memory captures PyTorch CUDA allocator behavior.

## Files Created

- src/estimators/combined_inference_estimator.py
- results/day17_combined_inference_predictions.csv
- results/day17_combined_inference_metrics.csv
- results/day17_combined_error_by_model.csv
- results/day17_combined_error_by_dtype.csv
- results/day17_estimator_comparison.csv
- report/day17_combined_inference_estimator.md
- plots/day17_allocated_actual_vs_predicted.png
- plots/day17_reserved_actual_vs_predicted.png
- plots/day17_estimator_mre_comparison.png
- plots/day17_combined_error_by_model.png

## Modules Combined

The CombinedInferenceEstimator uses:

1. ImprovedBaseMemoryEstimator
2. KVCacheEstimator
3. PrecisionAwareEstimator
4. AllocatorCorrectionEstimator

## What the Combined Estimator Does

The estimator takes:

- model_name
- batch_size
- input_tokens
- max_new_tokens
- dtype
- use_cache

and returns:

- predicted_peak_allocated_MB
- predicted_peak_reserved_MB
- base_prediction_MB
- estimated_kv_cache_MB
- dtype_scale
- precision_correction
- allocator_padding_MB
- parameter_memory_MB
- token_memory_component_MB
- batch_memory_component_MB
- model_size_overhead_MB

This makes the prediction pipeline cleaner and reusable.

## Global Metrics

| allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.204764 | 30.243674 | 69.797927 | 0.150284 | 4.545455 | 23.813088 | 53.846154 | 0.0 |

Rounded values:

- Allocated MRE: 4.20%
- Allocated mean error: 30.24%
- Allocated max error: 69.80%
- Allocated min error: 0.15%
- Reserved MRE: 4.55%
- Reserved mean error: 23.81%
- Reserved max error: 53.85%
- Reserved min error: 0.00%

## Error by Model

| model_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 28 | 2.364403 | 2.016841 | 8.995498 | 2.360899 | 2.197802 | 5.729167 | 325.885 | 325.306329 | 354.642857 | 354.071429 |
| sshleifer/tiny-gpt2 | 20 | 69.274654 | 69.183753 | 69.797927 | 53.846154 | 53.846154 | 53.846154 | 12.503 | 3.841592 | 26.000000 | 12.000000 |

## Main Model-wise Observation

The combined estimator performs well for distilgpt2.

For distilgpt2:

- allocated mean error = 2.36%
- allocated median error = 2.02%
- reserved mean error = 2.36%
- reserved median error = 2.20%

This is strong for the first combined inference estimator.

For tiny-gpt2:

- allocated mean error = 69.27%
- allocated median error = 69.18%
- reserved mean error = 53.85%
- reserved median error = 53.85%

tiny-gpt2 is still predicted poorly because it is extremely small and its memory is dominated by framework and allocator overhead.

This confirms the repeated finding that tiny-gpt2 is useful for pipeline validation but not ideal for realistic estimator evaluation.

## Error by dtype

| dtype | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 3 | 5.336898 | 5.336898 | 5.336898 | 1.052632 | 1.052632 | 1.052632 | 181.250000 | 171.576872 |
| fp32 | 45 | 31.904126 | 2.871159 | 69.797927 | 25.330452 | 5.729167 | 53.846154 | 196.246444 | 192.681743 |

## dtype Observation

For fp16 rows:

- allocated mean error = 5.34%
- reserved mean error = 1.05%

This is reasonable.

For fp32 rows, the mean error is high because fp32 includes all tiny-gpt2 rows.

However, fp32 median allocated error is only 2.87%, which shows that most fp32 rows are predicted reasonably.

The high fp32 mean error is mainly caused by tiny-gpt2.

## Estimator Comparison

| estimator | allocated_MRE | allocated_mean_error | allocated_max_error |
|---|---:|---:|---:|
| Day 13 Improved Base | 5.673501 | 30.371345 | 69.797927 |
| Day 14 Improved + KV | 3.752376 | 30.187126 | 69.797927 |
| Day 15 Precision Aware | 4.204764 | 30.243674 | 69.797927 |
| Day 17 Combined | 4.204764 | 30.243674 | 69.797927 |

## Estimator Comparison Interpretation

The Day 17 Combined estimator has the same allocated-memory metrics as the Day 15 PrecisionAwareEstimator.

This is expected because Day 17 integrates the same allocated-memory prediction path and adds reserved-memory prediction through allocator correction.

Compared to Day 14, the allocated MRE became slightly worse:

- Day 14 MRE = 3.75%
- Day 17 MRE = 4.20%

However, Day 17 is still important because it creates one complete reusable estimator pipeline that predicts both allocated and reserved memory.

The goal of Day 17 was integration, not aggressive tuning.

## Main Interpretation

Day 17 successfully combined the inference estimator modules into one pipeline.

The combined estimator now predicts both allocated and reserved memory.

The global allocated MRE is 4.20%, and the global reserved MRE is 4.55%.

The model-wise result is more important than the global mean.

For distilgpt2, the estimator performs well:

- allocated mean error = 2.36%
- reserved mean error = 2.36%

For tiny-gpt2, the estimator performs poorly because tiny-gpt2 is dominated by overhead and allocator effects.

This means final inference evaluation should report:

1. all rows
2. distilgpt2-only rows
3. tiny-gpt2 rows

This separation is necessary because tiny-gpt2 distorts global mean and max error.

## Why This Matters for the Project

This is a major milestone.

The project now has a reusable inference-side memory prediction system.

It includes:

- model-size-aware base prediction
- KV-cache memory estimation
- precision-aware adjustment
- allocator reserved-memory correction

This is much closer to the original goal of adapting xMem-style memory prediction to LLM workloads.

The estimator is not perfect, but it now has the right modular structure.

## Remaining Weaknesses

The combined estimator still has some issues:

1. tiny-gpt2 allocated memory is badly underpredicted.
2. tiny-gpt2 reserved memory is badly underpredicted.
3. allocator floor for very small models is missing.
4. precision correction is slightly too aggressive for fp16.
5. final MRE/PEF evaluation is still needed.
6. training memory is not included yet.

## Questions Answered

### Q1. Why do we need a combined estimator instead of separate modules?

Separate modules are useful during development, but a real prediction pipeline needs one reusable estimator.

The combined estimator takes model and workload inputs and returns predicted allocated and reserved memory using all modules together.

### Q2. What is the difference between predicted allocated memory and predicted reserved memory?

Predicted allocated memory estimates active tensor memory.

Predicted reserved memory estimates memory held by PyTorch's CUDA allocator, including extra cached blocks.

Reserved memory is closer to actual GPU memory pressure and OOM risk.

### Q3. Which modules are included in the CombinedInferenceEstimator?

The CombinedInferenceEstimator includes:

- ImprovedBaseMemoryEstimator
- KVCacheEstimator
- PrecisionAwareEstimator
- AllocatorCorrectionEstimator

### Q4. Did the combined estimator improve or worsen MRE compared to Day 14/15?

Compared to Day 14, allocated MRE became slightly worse:

- Day 14 MRE = 3.75%
- Day 17 MRE = 4.20%

Compared to Day 15, allocated MRE stayed the same:

- Day 15 MRE = 4.20%
- Day 17 MRE = 4.20%

But Day 17 adds reserved-memory prediction in one integrated pipeline.

So the main value of Day 17 is integration, not metric improvement.

### Q5. What is the next evaluation step after building the combined estimator?

The next step is final inference evaluation.

This should include:

- full estimator comparison
- all-row metrics
- distilgpt2-only metrics
- tiny-gpt2 metrics
- PEF-style simulation

## Day 17 Conclusion

On Day 17, I built the CombinedInferenceEstimator by integrating ImprovedBaseMemoryEstimator, KVCacheEstimator, PrecisionAwareEstimator, and AllocatorCorrectionEstimator.

The combined estimator predicts both peak allocated memory and peak reserved memory.

The global allocated MRE was 4.20%, mean allocated error was 30.24%, and max allocated error was 69.80%.

The global reserved MRE was 4.55%, mean reserved error was 23.81%, and max reserved error was 53.85%.

The model-wise results show the real picture.

For distilgpt2, the combined estimator performed well with 2.36% mean allocated error and 2.36% mean reserved error.

For tiny-gpt2, the estimator still performed poorly because tiny-gpt2 is dominated by framework and allocator overhead.

The key result is that the project now has a complete inference-side estimator pipeline that predicts both allocated and reserved memory.

The next step is to do final inference evaluation, separating realistic model results from toy-model results, and then run PEF-style simulation.

---


# Day 18 - Final Inference Evaluation

## Goal

Perform final inference-side evaluation of the estimator modules.

Day 18 focused on comparing all inference estimators and separating realistic-model metrics from toy-model metrics.

This was necessary because tiny-gpt2 strongly distorts global mean and max error due to framework and allocator overhead.

## Files Created

- results/day18_final_inference_estimator_comparison.csv
- results/day18_reserved_memory_summary.csv
- results/day18_key_findings.csv
- report/inference_phase_report.md
- plots/day18_all_rows_mre_comparison.png
- plots/day18_distilgpt2_mre_comparison.png
- plots/day18_all_vs_distilgpt2_mean_error.png
- plots/day18_combined_actual_vs_predicted_allocated.png
- plots/day18_combined_actual_vs_predicted_reserved.png

## Final Allocated Memory Comparison

| estimator | all_MRE | all_mean_error | all_max_error | distilgpt2_MRE | distilgpt2_mean_error | distilgpt2_max_error | tiny_gpt2_MRE | tiny_gpt2_mean_error | tiny_gpt2_max_error | num_rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day 11 Base | 6.817565 | 42.464030 | 95.138420 | 5.321602 | 5.282823 | 8.151976 | 95.138388 | 94.517720 | 95.138420 | 48 |
| Day 13 Improved Base | 5.673501 | 30.371345 | 69.797927 | 1.350309 | 2.570964 | 11.485668 | 69.193916 | 69.291879 | 69.797927 | 48 |
| Day 14 Improved + KV | 3.752376 | 30.187126 | 69.797927 | 2.016841 | 2.267462 | 8.995498 | 69.183753 | 69.274654 | 69.797927 | 48 |
| Day 15 Precision Aware | 4.204764 | 30.243674 | 69.797927 | 2.016841 | 2.364403 | 8.995498 | 69.183753 | 69.274654 | 69.797927 | 48 |
| Day 17 Combined | 4.204764 | 30.243674 | 69.797927 | 2.016841 | 2.364403 | 8.995498 | 69.183753 | 69.274654 | 69.797927 | 48 |

## Reserved Memory Summary

| subset | MRE | mean_error | max_error | num_rows |
|---|---:|---:|---:|---:|
| all_rows | 4.545455 | 23.813088 | 53.846154 | 48 |
| distilgpt2_only | 2.197802 | 2.360899 | 5.729167 | 28 |
| tiny_gpt2_only | 53.846154 | 53.846154 | 53.846154 | 20 |

## Key Findings

| finding | value | metric |
|---|---|---:|
| Best all-row allocated MRE | Day 14 Improved + KV | 3.752376 |
| Best distilgpt2 allocated mean error | Day 14 Improved + KV | 2.267462 |
| Day 17 distilgpt2 allocated mean error | CombinedInferenceEstimator | 2.364403 |
| Day 17 distilgpt2 reserved mean error | CombinedInferenceEstimator | 2.360899 |

## Main Finding

The best all-row allocated-memory MRE was achieved by the Day 14 Improved + KV estimator.

It achieved:

- all-row MRE = 3.75%

The best distilgpt2 allocated mean error was also achieved by Day 14 Improved + KV.

It achieved:

- distilgpt2 mean error = 2.27%

However, the Day 17 CombinedInferenceEstimator is the most complete estimator because it predicts both:

- peak allocated memory
- peak reserved memory

## CombinedInferenceEstimator Result

For distilgpt2, the CombinedInferenceEstimator achieved:

- allocated mean error = 2.36%
- reserved mean error = 2.36%

This is a strong inference-side result.

It means the combined estimator predicts distilgpt2 inference memory with around 2-3% mean error for both allocated and reserved memory.

## Why tiny-gpt2 Was Separated

tiny-gpt2 strongly distorts global mean and max error.

For tiny-gpt2:

- allocated mean error = 69.27%
- reserved mean error = 53.85%

This happens because tiny-gpt2 is extremely small.

Its actual memory is around:

- 12.5 MB allocated
- 26 MB reserved

At this scale, fixed framework overhead, CUDA overhead, and allocator behavior dominate the memory.

So small absolute prediction mistakes become very large relative errors.

Because of this, tiny-gpt2 should be treated mainly as a pipeline validation model, not as the main realistic evaluation target.

## Why distilgpt2 Metrics Matter More

distilgpt2 is larger and shows clearer memory-scaling behavior.

Earlier experiments showed that distilgpt2 responds meaningfully to:

- input token length
- generated token length
- batch size
- dtype
- cache setting
- allocator behavior

Therefore, distilgpt2-only metrics are more meaningful for judging whether the estimator works on realistic LLM-style workloads.

## Estimator Progression

The estimator evolved as follows:

### Day 11 Base Estimator

Used parameter memory, token memory, batch memory, and fixed overhead.

Problem:

- worked reasonably on distilgpt2
- failed badly on tiny-gpt2 because fixed overhead was too large

### Day 13 Improved Base Estimator

Replaced fixed overhead with model-size-aware overhead.

Result:

- improved global MRE from 6.82% to 5.67%
- improved distilgpt2 mean error from 5.28% to 2.57%

### Day 14 Improved + KV

Added explicit KV-cache estimation.

Result:

- best all-row MRE = 3.75%
- best distilgpt2 allocated mean error = 2.27%

### Day 15 Precision Aware

Added dtype-aware memory adjustment.

Result:

- fp16 prediction error was around 5.34%
- global MRE did not improve because only 3 rows were fp16

### Day 17 CombinedInferenceEstimator

Combined base, KV, precision, and allocator modules.

Result:

- allocated MRE = 4.20%
- reserved MRE = 4.55%
- distilgpt2 allocated mean error = 2.36%
- distilgpt2 reserved mean error = 2.36%

## Main Interpretation

The all-row metrics are useful, but they are distorted by tiny-gpt2.

The distilgpt2-only metrics show that the estimator is working well for the more realistic model.

The CombinedInferenceEstimator is slightly worse than Day 14 for allocated-memory MRE, but it is more complete because it predicts both allocated and reserved memory.

Therefore, the final inference-phase conclusion should not only focus on the lowest MRE.

It should say:

- Day 14 gave the best pure allocated-memory MRE.
- Day 17 gave the most complete reusable estimator pipeline.
- distilgpt2 results are the most meaningful for realistic LLM memory prediction.

## Inference Phase Result

The main inference-phase result is:

The project now has a modular inference memory prediction pipeline that predicts distilgpt2 inference memory with around 2-3% mean error for both allocated and reserved memory.

This is a strong result for the inference phase.

## Questions Answered

### Q1. Why should we report distilgpt2-only metrics separately?

Because tiny-gpt2 is too small and its memory is dominated by PyTorch/framework/allocator overhead.

It distorts global mean and max error.

distilgpt2 is more realistic for evaluating LLM memory prediction.

### Q2. Which estimator had the best all-row MRE?

Day 14 Improved + KV had the best all-row MRE.

Its all-row MRE was:

3.75%

### Q3. How well did the CombinedInferenceEstimator perform on distilgpt2?

It performed well.

For distilgpt2:

- allocated mean error = 2.36%
- reserved mean error = 2.36%

This is strong for a first combined inference estimator.

### Q4. Why does tiny-gpt2 distort global mean and max error?

Because its actual memory is very small.

It uses around:

- 12.5 MB allocated memory
- 26 MB reserved memory

At this scale, fixed overhead and allocator behavior become large relative to the actual model memory.

So even small absolute prediction errors become huge percentage errors.

### Q5. What is the next step after final inference evaluation?

The next step is PEF-style simulation.

This will test whether the estimator correctly predicts fit/fail behavior under different GPU memory limits.

## Day 18 Conclusion

Day 18 completed the final inference-side evaluation of the estimator modules.

The best all-row allocated-memory MRE was achieved by the Day 14 Improved + KV estimator, with an MRE of 3.75%.

The best distilgpt2 allocated mean error was also achieved by Day 14 Improved + KV, with 2.27% mean error.

The Day 17 CombinedInferenceEstimator had a slightly higher allocated MRE of 4.20%, but it is the most complete pipeline because it predicts both peak allocated and peak reserved memory.

For distilgpt2, the CombinedInferenceEstimator achieved 2.36% mean allocated error and 2.36% mean reserved error.

The evaluation also showed that tiny-gpt2 strongly distorts global mean and max error.

tiny-gpt2 had around 69.27% allocated mean error and 53.85% reserved mean error because its memory is dominated by framework and allocator overhead.

Therefore, tiny-gpt2 should be treated as a pipeline validation model, while distilgpt2 is the more meaningful model for inference estimator evaluation.

The main inference-phase result is that the project now has a modular estimator pipeline that predicts distilgpt2 inference memory with around 2-3% mean error for both allocated and reserved memory.

The next step is PEF-style simulation.

---


# Day 19 - PEF-style Simulation

## Goal

Run a PEF-style fit/fail simulation using the CombinedInferenceEstimator.

The goal was to test whether the estimator can correctly predict if a workload fits inside a given GPU memory limit.

Until now, the project focused mainly on numeric memory prediction error.

Day 19 checked scheduling-style behavior:

- Will the workload fit?
- Will the estimator wrongly allow a workload that actually exceeds memory?
- Will the estimator wrongly reject a workload that actually fits?

This is important because memory prediction is useful only if it can help avoid OOM failures.

## Files Created

- results/day19_pef_simulation.csv
- results/day19_pef_summary.csv
- results/day19_pef_by_model.csv
- results/day19_pef_by_memory_limit.csv
- results/day19_dangerous_failures.csv
- results/day19_conservative_failures.csv
- results/day19_realistic_gpu_limit_summary.csv
- report/day19_pef_style_simulation.md
- plots/day19_pef_failure_rate_by_limit.png
- plots/day19_pef_accuracy_by_model.png
- plots/day19_dangerous_failure_by_model.png

## Memory Used for Fit/Fail Decision

The simulation used peak reserved memory.

Actual memory:

actual_peak_reserved_MB

Predicted memory:

predicted_peak_reserved_MB

Reserved memory was used instead of allocated memory because reserved memory better reflects PyTorch CUDA allocator behavior and real GPU memory pressure.

## Definition Used

### Dangerous Failure

A dangerous failure happens when:

predicted_reserved_MB <= memory_limit_MB

but

actual_reserved_MB > memory_limit_MB

This means the estimator predicts that the workload will fit, but it actually exceeds the memory limit.

This is dangerous because it can cause an OOM failure.

### Conservative Failure

A conservative failure happens when:

predicted_reserved_MB > memory_limit_MB

but

actual_reserved_MB <= memory_limit_MB

This means the estimator predicts that the workload will not fit, but it actually would fit.

This wastes GPU capacity, but it is safer than dangerous underprediction.

## Memory Limits Tested

The following memory limits were tested:

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

Small artificial limits were used to test boundary behavior.

Realistic limits like 4096 MB and 8192 MB were included for discussion.

## Overall PEF Summary

| total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|
| 720 | 678 | 40 | 2 | 94.166667 | 5.555556 | 0.277778 |

## Overall Result

Across 720 simulated fit/fail cases:

- Correct predictions: 678
- Dangerous failures: 40
- Conservative failures: 2
- Accuracy: 94.17%
- Dangerous failure rate: 5.56%
- Conservative failure rate: 0.28%

This shows that the estimator is mostly correct in fit/fail prediction.

However, the dangerous failures were not random. They came from tiny-gpt2 at very small artificial memory limits.

## PEF by Model

| model_name | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 420 | 418 | 0 | 2 | 99.523810 | 0.000000 | 0.476190 |
| sshleifer/tiny-gpt2 | 300 | 260 | 40 | 0 | 86.666667 | 13.333333 | 0.000000 |

## distilgpt2 Result

For distilgpt2, the estimator performed very well.

- Accuracy: 99.52%
- Dangerous failures: 0
- Dangerous failure rate: 0.00%
- Conservative failures: 2
- Conservative failure rate: 0.48%

This is a strong result for scheduling-style prediction.

The estimator did not dangerously underpredict any distilgpt2 case.

The only failures were conservative failures, where the estimator predicted slightly more memory than actual.

This is safer than underprediction.

## tiny-gpt2 Result

For tiny-gpt2, the estimator performed worse.

- Accuracy: 86.67%
- Dangerous failures: 40
- Dangerous failure rate: 13.33%
- Conservative failures: 0

The dangerous failures happened because tiny-gpt2 actual reserved memory was 26 MB, while predicted reserved memory was around 12 MB.

So at artificial memory limits of 16 MB and 24 MB, the estimator predicted that tiny-gpt2 would fit, but actual reserved memory exceeded the limit.

This again shows that tiny-gpt2 needs a reserved-memory floor or special handling.

## PEF by Memory Limit

| memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 48 | 28 | 20 | 0 | 58.333333 | 41.666667 | 0.000000 |
| 24 | 48 | 28 | 20 | 0 | 58.333333 | 41.666667 | 0.000000 |
| 26 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 32 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 64 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 128 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 190 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 256 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 360 | 48 | 46 | 0 | 2 | 95.833333 | 0.000000 | 4.166667 |
| 400 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 512 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 1024 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 2048 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 4096 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 8192 | 48 | 48 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |

## Memory-limit Observation

At 16 MB and 24 MB:

- dangerous failure rate = 41.67%

These failures came from tiny-gpt2.

At 26 MB and above, tiny-gpt2 fits correctly because its actual reserved memory is 26 MB.

At 360 MB:

- conservative failure rate = 4.17%

These were distilgpt2 cases where predicted memory was slightly higher than actual memory.

The conservative failures were:

- actual reserved memory = 358 MB, predicted = 370 MB
- actual reserved memory = 360 MB, predicted = 372 MB

This is not dangerous. It means the estimator was slightly conservative near the boundary.

## Dangerous Failures

The dangerous failures came from tiny-gpt2.

Example pattern:

| model_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB |
|---|---:|---:|---:|
| sshleifer/tiny-gpt2 | 16 | 26.0 | 12.0 |
| sshleifer/tiny-gpt2 | 24 | 26.0 | 12.0 |

Reason:

The estimator underpredicted tiny-gpt2 reserved memory because it did not include a reserved-memory floor for very small models.

## Conservative Failures

There were 2 conservative failures.

Both were from distilgpt2 at 360 MB memory limit:

| model_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB |
|---|---:|---:|---:|
| distilgpt2 | 360 | 358.0 | 370.0 |
| distilgpt2 | 360 | 360.0 | 372.0 |

These are safer than dangerous failures because the estimator would reject workloads that actually fit, instead of allowing workloads that OOM.

## Realistic GPU Limit Summary

| memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent |
|---:|---:|---:|---:|---:|---:|
| 4096 | 48 | 48 | 0 | 0 | 100.0 |
| 8192 | 48 | 48 | 0 | 0 | 100.0 |

## Realistic Limit Interpretation

At realistic GPU limits like 4 GB and 8 GB, all workloads were correctly predicted to fit.

This is expected because the tested models are small compared to 4 GB and 8 GB GPUs.

The artificial smaller limits were necessary to test boundary behavior.

## Main Interpretation

The PEF-style simulation shows that the estimator is useful for scheduling-style fit/fail prediction.

Overall accuracy was 94.17%.

For distilgpt2, accuracy was 99.52%, with 0 dangerous failures.

This is the most important result because distilgpt2 is the more realistic model in the current experiments.

The main weakness is tiny-gpt2 at very small limits.

This confirms the earlier conclusion that tiny-gpt2 is useful for pipeline validation but not reliable as the main memory-prediction benchmark.

## What This Means for the Project

Day 19 adds an important evaluation dimension.

The project now evaluates:

- numeric memory prediction error
- allocated memory MRE
- reserved memory MRE
- fit/fail prediction accuracy
- dangerous underprediction rate
- conservative overprediction rate

This makes the project stronger because memory prediction is not only about reducing relative error.

It is also about avoiding wrong scheduling decisions that cause OOM.

## Remaining Weaknesses

1. tiny-gpt2 still needs a reserved-memory floor.
2. Current workloads are too small to stress 4 GB or 8 GB GPUs.
3. Larger models will be needed later for realistic GPU-limit stress testing.
4. PEF-style simulation is currently based on observed small-model workloads.
5. Training workloads are not included yet.

## Questions Answered

### Q1. What is a dangerous failure in PEF-style evaluation?

A dangerous failure happens when the estimator predicts that a workload will fit, but the actual memory exceeds the limit.

Formula:

predicted_reserved_MB <= memory_limit_MB

and

actual_reserved_MB > memory_limit_MB

This is dangerous because it can cause OOM.

### Q2. What is a conservative failure?

A conservative failure happens when the estimator predicts that a workload will not fit, but the actual memory would fit.

Formula:

predicted_reserved_MB > memory_limit_MB

and

actual_reserved_MB <= memory_limit_MB

This wastes GPU capacity but avoids OOM.

### Q3. Why did we use reserved memory instead of allocated memory?

Reserved memory better reflects PyTorch CUDA allocator behavior and real GPU memory pressure.

OOM risk depends on more than active tensor memory.

So reserved memory is safer for fit/fail simulation.

### Q4. What was the dangerous failure rate?

Overall dangerous failure rate:

5.56%

For distilgpt2:

0.00%

For tiny-gpt2:

13.33%

### Q5. Why do realistic 4 GB / 8 GB limits not stress current workloads?

Because the current models are small.

The largest reserved memory in the current experiments is far below 4096 MB.

So all workloads easily fit under 4 GB and 8 GB limits.

Artificial smaller limits are needed to test boundary behavior with the current dataset.

## Day 19 Conclusion

On Day 19, I performed a PEF-style fit/fail simulation using predicted and actual peak reserved memory.

Across 720 simulated cases, the estimator achieved 94.17% fit/fail accuracy.

The dangerous failure rate was 5.56%, and the conservative failure rate was 0.28%.

For distilgpt2, the estimator performed very well with 99.52% accuracy and 0% dangerous failure rate.

The only distilgpt2 failures were 2 conservative failures at the 360 MB memory limit, where the estimator overpredicted memory slightly.

This is safer than underprediction because it would reject a workload that could actually fit instead of causing OOM.

For tiny-gpt2, the estimator had a 13.33% dangerous failure rate because it predicted reserved memory around 12 MB while actual reserved memory was 26 MB.

These failures occurred at artificial memory limits of 16 MB and 24 MB.

This confirms that tiny-gpt2 needs a reserved-memory floor or special handling because its memory is dominated by PyTorch allocator overhead.

For realistic GPU limits such as 4 GB and 8 GB, all workloads were correctly predicted to fit, giving 100% accuracy.

The main conclusion is that the estimator is reliable for distilgpt2 scheduling-style decisions, but tiny toy models can cause dangerous underprediction unless allocator floor behavior is handled.

The next step is GitHub repository setup and clean project structure.

---


# Day 21 - Training Memory Logger

## Goal

Start the training-memory phase by building a reusable training memory logger.

Until Day 20, the project focused on inference profiling, inference estimators, PEF-style simulation, and GitHub setup.

Day 21 created and tested the first training memory logger.

The goal was not to run many experiments. The goal was to confirm that the logger correctly captures memory across the main training stages.

## Files Created

- src/training_logger.py
- results/training_runs.csv
- results/day21_tiny_gpt2_training_stage_breakdown.csv
- results/day21_training_vs_inference_tiny_gpt2.csv
- report/day21_training_logger.md
- plots/day21_training_allocated_by_stage.png
- plots/day21_training_peak_memory_by_stage.png

## Logger Stages

The training logger records memory at the following stages:

- before_model_load
- after_model_load
- after_batch_creation
- after_optimizer_creation
- after_forward
- after_loss
- after_backward
- after_optimizer_step
- after_zero_grad

For each stage, it records:

- allocated_MB
- reserved_MB
- peak_allocated_MB
- peak_reserved_MB

## First Training Test

Model:

sshleifer/tiny-gpt2

Settings:

- batch_size = 1
- input_tokens = 32
- dtype = fp32
- optimizer = AdamW

## Stage Breakdown

| stage | allocated_MB | reserved_MB | peak_allocated_MB | peak_reserved_MB |
|---|---:|---:|---:|---:|
| before_model_load | 0.00 | 0.0 | 0.00 | 0.0 |
| after_model_load | 2.79 | 4.0 | 2.79 | 4.0 |
| after_batch_creation | 2.79 | 4.0 | 2.79 | 4.0 |
| after_optimizer_creation | 2.79 | 4.0 | 2.79 | 4.0 |
| after_forward | 24.21 | 44.0 | 24.21 | 44.0 |
| after_loss | 24.21 | 44.0 | 24.21 | 44.0 |
| after_backward | 26.97 | 46.0 | 36.48 | 46.0 |
| after_optimizer_step | 28.54 | 48.0 | 36.48 | 48.0 |
| after_zero_grad | 27.75 | 48.0 | 36.48 | 48.0 |

## Memory Difference by Stage

| stage | allocated_MB | allocated_diff_MB | reserved_MB | reserved_diff_MB | peak_allocated_MB | peak_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|
| before_model_load | 0.00 | NaN | 0.0 | NaN | 0.00 | 0.0 |
| after_model_load | 2.79 | 2.79 | 4.0 | 4.0 | 2.79 | 4.0 |
| after_batch_creation | 2.79 | 0.00 | 4.0 | 0.0 | 2.79 | 4.0 |
| after_optimizer_creation | 2.79 | 0.00 | 4.0 | 0.0 | 2.79 | 4.0 |
| after_forward | 24.21 | 21.42 | 44.0 | 40.0 | 24.21 | 44.0 |
| after_loss | 24.21 | 0.00 | 44.0 | 0.0 | 24.21 | 44.0 |
| after_backward | 26.97 | 2.76 | 46.0 | 2.0 | 36.48 | 46.0 |
| after_optimizer_step | 28.54 | 1.57 | 48.0 | 2.0 | 36.48 | 48.0 |
| after_zero_grad | 27.75 | -0.79 | 48.0 | 0.0 | 36.48 | 48.0 |

## Main Results

The highest peak allocated memory was:

36.48 MB

The highest peak reserved memory was:

48.0 MB

The biggest current allocated-memory jump happened after the forward pass:

2.79 MB to 24.21 MB

This was an increase of:

21.42 MB

The biggest reserved-memory jump also happened after the forward pass:

4.0 MB to 44.0 MB

This was an increase of:

40.0 MB

## Stage-wise Interpretation

### Before Model Load

Allocated and reserved memory were both zero.

This was the clean starting point after memory cleanup.

### After Model Load

Allocated memory became 2.79 MB.

Reserved memory became 4.0 MB.

This represents model weights and basic CUDA/PyTorch memory.

Since tiny-gpt2 is very small, this value is also small.

### After Batch Creation

Allocated memory stayed at 2.79 MB.

Reserved memory stayed at 4.0 MB.

The input batch was too small to create a visible memory change.

### After Optimizer Creation

Allocated memory stayed at 2.79 MB.

Reserved memory stayed at 4.0 MB.

This is expected because AdamW optimizer states are usually created lazily during the first optimizer step, not immediately at optimizer creation.

### After Forward Pass

Allocated memory increased from 2.79 MB to 24.21 MB.

Reserved memory increased from 4.0 MB to 44.0 MB.

This was the biggest visible memory jump.

Reason:

Training stores activations during the forward pass because they are needed later for backward computation.

This is one of the key differences between training and inference.

### After Loss

Allocated memory stayed at 24.21 MB.

Reserved memory stayed at 44.0 MB.

The loss tensor itself is very small, so it did not create a meaningful memory increase.

### After Backward

Allocated memory increased from 24.21 MB to 26.97 MB.

Reserved memory increased from 44.0 MB to 46.0 MB.

Peak allocated memory increased to 36.48 MB.

This is important because the current allocated memory after backward was 26.97 MB, but the peak allocated memory reached 36.48 MB during backward.

This means backward created temporary tensors that increased peak memory, but some of them were released before the snapshot was taken.

This shows why peak memory is more important than final allocated memory.

### After Optimizer Step

Allocated memory increased to 28.54 MB.

Reserved memory increased to 48.0 MB.

This slight increase likely comes from optimizer state creation/update.

For AdamW, optimizer states matter because AdamW stores first and second moment tensors.

### After zero_grad

Allocated memory decreased from 28.54 MB to 27.75 MB.

Reserved memory stayed at 48.0 MB.

This confirms PyTorch allocator behavior.

Active tensor memory can decrease, but reserved memory can stay cached for reuse.

## Main Interpretation

The training logger is working correctly.

It successfully captured memory across:

- model loading
- batch creation
- optimizer creation
- forward pass
- loss computation
- backward pass
- optimizer step
- zero_grad

The forward pass caused the largest visible memory jump because activations are stored for backward.

Backward caused the highest peak allocated memory because it creates gradients and temporary backward tensors.

After zero_grad, allocated memory decreased slightly, but reserved memory stayed the same due to PyTorch CUDA allocator caching.

This confirms that training memory is more complex than inference memory.

## Why This Matters for the Project

The project now has the foundation for training-side memory profiling.

Until now, the estimator work focused mainly on inference.

Training memory requires additional components:

- activations
- gradients
- optimizer states
- backward temporary tensors
- optimizer-step behavior

Day 21 starts this second major phase of the project.

## Questions Answered

### Q1. Why is training memory usually higher than inference memory?

Training memory is usually higher because training stores activations for backward, gradients for parameters, and optimizer states.

Inference mostly performs forward computation and can discard many intermediate tensors.

### Q2. Which stage used the highest peak memory?

The highest peak allocated memory was 36.48 MB.

It appeared during/after backward.

The highest peak reserved memory was 48.0 MB.

It appeared after optimizer step and stayed after zero_grad.

### Q3. What changed after backward?

After backward:

- allocated memory increased from 24.21 MB to 26.97 MB
- reserved memory increased from 44.0 MB to 46.0 MB
- peak allocated memory increased to 36.48 MB

This shows that gradients and temporary backward tensors increased memory.

The peak was higher than the current snapshot because backward creates temporary tensors that may be freed before the snapshot.

### Q4. Why does optimizer choice matter for training memory?

Optimizer choice matters because optimizers can store extra state.

For example:

- SGD without momentum stores little or no extra state.
- SGD with momentum stores one extra tensor per parameter.
- AdamW stores two extra tensors per parameter: first moment and second moment.

Therefore, AdamW usually uses more memory than simple SGD.

### Q5. Why did we start with tiny-gpt2 instead of distilgpt2?

Training is more memory-heavy and more error-prone than inference.

tiny-gpt2 is small, so it is safer for debugging the logger.

Once the logger is confirmed to work, the project can move to distilgpt2 training experiments.

## Day 21 Conclusion

Day 21 successfully built the training memory logger and tested it on sshleifer/tiny-gpt2.

The logger captured memory across all important training stages.

The highest peak allocated memory was 36.48 MB.

The highest peak reserved memory was 48.0 MB.

The biggest memory jump happened after the forward pass because activations were stored for backward computation.

The backward stage created the highest peak allocated memory, showing that peak memory can be higher than final snapshot memory.

After zero_grad, allocated memory decreased slightly, but reserved memory stayed the same due to PyTorch allocator caching.

The training logger is now ready for controlled training experiments.

The next step is Day 22: run controlled tiny-gpt2 training sanity experiments before moving to distilgpt2.

---


# Day 22 - tiny-gpt2 Training Sanity Experiments

## Goal

Run controlled tiny-gpt2 training experiments to verify that the training memory logger behaves correctly.

Day 21 tested one training run.

Day 22 tested the logger under:

- sequence-length variation
- optimizer variation

The goal was not to make final training conclusions. The goal was to confirm that the logger is stable before moving to distilgpt2.

## Files Created

- results/day22_tiny_training_sequence.csv
- results/day22_tiny_training_sequence_stages.csv
- results/day22_tiny_training_optimizer.csv
- results/day22_tiny_training_optimizer_stages.csv
- results/day22_optimizer_stage_pivot.csv
- results/day22_tiny_training_summary.csv
- report/day22_tiny_training_sanity.md
- plots/day22_tiny_training_memory_vs_sequence.png
- plots/day22_tiny_optimizer_peak_allocated.png
- plots/day22_tiny_optimizer_peak_reserved.png

## Experiment 1 - Sequence-length Variation

Settings:

- model = sshleifer/tiny-gpt2
- batch_size = 1
- dtype = fp32
- optimizer = AdamW

Input token lengths tested:

- 16
- 32
- 64
- 128

## Sequence-length Results

| model_name | batch_size | input_tokens | optimizer_name | dtype | loss | runtime_sec | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | oom |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| sshleifer/tiny-gpt2 | 1 | 16 | adamw | fp32 | 10.828448 | 5.676 | 27.27 | 48.0 | 25.37 | 48.0 | False |
| sshleifer/tiny-gpt2 | 1 | 32 | adamw | fp32 | 10.829083 | 1.060 | 45.29 | 68.0 | 28.44 | 68.0 | False |
| sshleifer/tiny-gpt2 | 1 | 64 | adamw | fp32 | 10.830624 | 1.043 | 69.84 | 84.0 | 34.57 | 84.0 | False |
| sshleifer/tiny-gpt2 | 1 | 128 | adamw | fp32 | 10.827228 | 1.057 | 118.95 | 132.0 | 46.84 | 132.0 | False |

## Sequence-length Observation

Peak allocated memory increased clearly with input sequence length:

- 16 tokens: 27.27 MB
- 32 tokens: 45.29 MB
- 64 tokens: 69.84 MB
- 128 tokens: 118.95 MB

Peak reserved memory also increased:

- 16 tokens: 48.0 MB
- 32 tokens: 68.0 MB
- 64 tokens: 84.0 MB
- 128 tokens: 132.0 MB

This confirms that training memory scales clearly with sequence length.

This happens because training stores activations for backward computation, and longer sequences create more activations.

## Runtime Observation

The first run with 16 tokens took 5.676 seconds, while the later runs took around 1 second.

This does not mean 16 tokens is slower.

The first run likely included warmup overhead such as:

- model/tokenizer loading
- CUDA initialization
- kernel warmup
- Hugging Face cache setup

So runtime should not be compared directly unless repeated runs are averaged.

## Experiment 2 - Optimizer Comparison

Settings:

- model = sshleifer/tiny-gpt2
- batch_size = 1
- input_tokens = 64
- dtype = fp32

Optimizers tested:

- SGD
- AdamW

## Optimizer Results

| model_name | batch_size | input_tokens | optimizer_name | dtype | loss | runtime_sec | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | oom |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| sshleifer/tiny-gpt2 | 1 | 64 | sgd | fp32 | 10.831019 | 1.166 | 69.84 | 82.0 | 32.99 | 82.0 | False |
| sshleifer/tiny-gpt2 | 1 | 64 | adamw | fp32 | 10.830429 | 1.035 | 69.84 | 84.0 | 34.57 | 84.0 | False |

## Optimizer Observation

Peak allocated memory was the same for SGD and AdamW:

- SGD peak allocated = 69.84 MB
- AdamW peak allocated = 69.84 MB

However, AdamW used slightly more reserved and final allocated memory:

- SGD final allocated = 32.99 MB
- AdamW final allocated = 34.57 MB

- SGD peak reserved = 82.0 MB
- AdamW peak reserved = 84.0 MB

This suggests that AdamW has slightly higher optimizer-state overhead, but tiny-gpt2 is too small to show a large difference.

For larger models like distilgpt2, optimizer-state memory should become more visible.

## Optimizer Stage Breakdown

| optimizer_name | stage | allocated_MB | reserved_MB | peak_allocated_MB | peak_reserved_MB |
|---|---|---:|---:|---:|---:|
| sgd | before_model_load | 17.93 | 22.0 | 17.93 | 22.0 |
| sgd | after_model_load | 20.72 | 24.0 | 20.72 | 24.0 |
| sgd | after_batch_creation | 20.72 | 24.0 | 20.72 | 24.0 |
| sgd | after_optimizer_creation | 20.72 | 24.0 | 20.72 | 24.0 |
| sgd | after_forward | 45.30 | 52.0 | 45.30 | 52.0 |
| sgd | after_loss | 45.30 | 52.0 | 45.30 | 52.0 |
| sgd | after_backward | 33.78 | 82.0 | 69.84 | 82.0 |
| sgd | after_optimizer_step | 33.78 | 82.0 | 69.84 | 82.0 |
| sgd | after_zero_grad | 32.99 | 82.0 | 69.84 | 82.0 |
| adamw | before_model_load | 17.93 | 22.0 | 17.93 | 22.0 |
| adamw | after_model_load | 20.72 | 24.0 | 20.72 | 24.0 |
| adamw | after_batch_creation | 20.72 | 24.0 | 20.72 | 24.0 |
| adamw | after_optimizer_creation | 20.72 | 24.0 | 20.72 | 24.0 |
| adamw | after_forward | 45.30 | 52.0 | 45.30 | 52.0 |
| adamw | after_loss | 45.30 | 52.0 | 45.30 | 52.0 |
| adamw | after_backward | 33.78 | 82.0 | 69.84 | 82.0 |
| adamw | after_optimizer_step | 35.36 | 84.0 | 69.84 | 84.0 |
| adamw | after_zero_grad | 34.57 | 84.0 | 69.84 | 84.0 |

## Stage Pivot

| stage | adamw | sgd |
|---|---:|---:|
| after_backward | 69.84 | 69.84 |
| after_batch_creation | 20.72 | 20.72 |
| after_forward | 45.30 | 45.30 |
| after_loss | 45.30 | 45.30 |
| after_model_load | 20.72 | 20.72 |
| after_optimizer_creation | 20.72 | 20.72 |
| after_optimizer_step | 69.84 | 69.84 |
| after_zero_grad | 69.84 | 69.84 |
| before_model_load | 17.93 | 17.93 |

## Summary

| experiment | num_runs | min_peak_allocated_MB | max_peak_allocated_MB | min_peak_reserved_MB | max_peak_reserved_MB | oom_count |
|---|---:|---:|---:|---:|---:|---:|
| sequence_length_variation | 4 | 27.27 | 118.95 | 48.0 | 132.0 | 0 |
| optimizer_comparison | 2 | 69.84 | 69.84 | 82.0 | 84.0 | 0 |

## Important Cleanup Observation

In the optimizer comparison, memory before model load was not zero:

before_model_load = 17.93 MB allocated

This means the runtime was not perfectly clean before the optimizer comparison.

This is not fatal for Day 22 because the goal was sanity testing.

However, for distilgpt2 training experiments, stricter cleanup is needed.

For future runs, the project should:

- restart runtime before major experiments
- call cleanup_memory before every run
- record before_model_load memory
- avoid overinterpreting first-run runtime

## Main Interpretation

The training logger is stable enough to move forward.

The sequence-length experiment showed a clear increase in training memory as input tokens increased.

This is important because training memory is more sensitive to sequence length than tiny-gpt2 inference memory was.

The optimizer comparison showed that AdamW uses slightly more reserved/final memory than SGD, but tiny-gpt2 is too small to show a major optimizer-state difference.

The next meaningful experiment should use distilgpt2.

## Why This Matters for the Project

Day 22 confirms that training memory can now be measured in a controlled way.

The project can now move from logger testing to real training memory profiling.

The next phase should focus on distilgpt2 because it is large enough to show clearer memory trends.

## Questions Answered

### Q1. Did peak memory increase with input sequence length?

Yes.

Peak allocated memory increased from 27.27 MB at 16 tokens to 118.95 MB at 128 tokens.

Peak reserved memory increased from 48.0 MB to 132.0 MB.

### Q2. Was the trend smooth or noisy?

The memory trend was mostly smooth and increasing.

The runtime trend was noisy because the first run had warmup overhead.

### Q3. Which optimizer used more memory: SGD or AdamW?

AdamW used slightly more reserved and final allocated memory.

Peak allocated memory was the same for both optimizers at 69.84 MB.

But AdamW had higher peak reserved memory:

- SGD = 82.0 MB
- AdamW = 84.0 MB

### Q4. Why might AdamW use more memory than SGD?

AdamW stores optimizer states, usually first moment and second moment tensors for parameters.

SGD without momentum has much less optimizer-state memory.

Therefore, AdamW usually uses more memory than simple SGD.

### Q5. Is tiny-gpt2 enough for final training conclusions?

No.

tiny-gpt2 is useful for sanity testing, but it is too small for final training-memory conclusions.

For final conclusions, distilgpt2 and possibly gpt2 are needed.

## Day 22 Conclusion

Day 22 completed controlled tiny-gpt2 training sanity experiments.

The sequence-length experiment showed that training memory increases clearly with input sequence length.

Peak allocated memory increased from 27.27 MB to 118.95 MB.

Peak reserved memory increased from 48.0 MB to 132.0 MB.

The optimizer comparison showed that AdamW used slightly more reserved and final allocated memory than SGD, but the peak allocated memory was the same for this tiny model.

The first run showed warmup overhead, and the optimizer comparison showed non-zero memory before model load, so future distilgpt2 runs need stricter cleanup or runtime restart.

Overall, the training logger is stable enough to move to distilgpt2 training sequence-length experiments.

The next step is Day 23: distilgpt2 training sequence-length experiment.

---


# Day 23 - distilgpt2 Training Sequence-length Experiment

## Goal

Run the first meaningful training-memory experiment using distilgpt2.

Day 21 built the training memory logger.

Day 22 verified the logger using controlled tiny-gpt2 training experiments.

Day 23 moved to distilgpt2 to observe more realistic training-memory behavior.

## Files Created

- results/day23_distilgpt2_training_sequence.csv
- results/day23_distilgpt2_training_sequence_stages.csv
- results/day23_distilgpt2_stage_peak_allocated_pivot.csv
- results/day23_distilgpt2_stage_peak_reserved_pivot.csv
- results/day23_distilgpt2_training_sequence_scaling.csv
- results/day23_distilgpt2_training_vs_inference.csv
- results/day23_distilgpt2_training_sequence_summary.csv
- report/day23_distilgpt2_training_sequence.md
- plots/day23_distilgpt2_training_memory_vs_sequence.png
- plots/day23_distilgpt2_allocated_by_stage.png
- plots/day23_distilgpt2_peak_allocated_by_stage.png

## Experiment Settings

- model = distilgpt2
- task = training
- batch_size = 1
- dtype = fp32
- optimizer = AdamW
- learning_rate = 5e-5
- GPU = Tesla T4

Input token lengths tested:

- 32
- 64
- 128

## Main Results

| input_tokens | loss | runtime_sec | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | oom |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 3.815424 | 9.323 | 1592.43 | 1700.0 | 965.34 | 1700.0 | False |
| 64 | 2.154191 | 1.600 | 1600.69 | 1738.0 | 972.73 | 1738.0 | False |
| 128 | 0.969172 | 1.549 | 1616.71 | 1726.0 | 987.62 | 1726.0 | False |

## Main Memory Trend

Peak allocated memory increased with sequence length:

- 32 tokens: 1592.43 MB
- 64 tokens: 1600.69 MB
- 128 tokens: 1616.71 MB

From 32 tokens to 128 tokens, peak allocated memory increased by:

24.28 MB

This shows that training memory increases with sequence length, but the increase is relatively small compared to the large fixed training-memory components.

Peak reserved memory stayed in the range:

1700 MB to 1738 MB

Reserved memory did not increase smoothly because PyTorch CUDA allocator reserves memory in chunks.

## Scaling Analysis

| input_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom | allocated_per_token_MB | reserved_per_token_MB |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 32 | 1592.43 | 1700.0 | 965.34 | 1700.0 | 9.323 | False | 49.763438 | 53.125000 |
| 64 | 1600.69 | 1738.0 | 972.73 | 1738.0 | 1.600 | False | 25.010781 | 27.156250 |
| 128 | 1616.71 | 1726.0 | 987.62 | 1726.0 | 1.549 | False | 12.630547 | 13.484375 |

## Per-token Interpretation

Allocated memory per token decreased as sequence length increased.

This does not mean longer sequences are cheaper.

It means there is a large fixed memory cost from:

- model parameters
- gradients
- optimizer states
- CUDA/PyTorch overhead
- allocator-reserved memory

Because this fixed cost is large, dividing by more tokens reduces the per-token average.

## Stage Peak Allocated Pivot

| stage | 32 tokens | 64 tokens | 128 tokens |
|---|---:|---:|---:|
| before_model_load | 0.00 | 17.88 | 17.88 |
| after_model_load | 313.23 | 331.73 | 331.73 |
| after_batch_creation | 313.24 | 331.74 | 331.74 |
| after_optimizer_creation | 313.24 | 331.74 | 331.74 |
| after_forward | 352.65 | 391.04 | 456.09 |
| after_loss | 352.65 | 391.04 | 456.09 |
| after_backward | 948.23 | 956.99 | 972.64 |
| after_optimizer_step | 1592.43 | 1600.69 | 1616.71 |
| after_zero_grad | 1592.43 | 1600.69 | 1616.71 |

## Stage Peak Reserved Pivot

| stage | 32 tokens | 64 tokens | 128 tokens |
|---|---:|---:|---:|
| before_model_load | 0.0 | 42.0 | 42.0 |
| after_model_load | 350.0 | 350.0 | 350.0 |
| after_batch_creation | 350.0 | 350.0 | 350.0 |
| after_optimizer_creation | 350.0 | 350.0 | 350.0 |
| after_forward | 388.0 | 416.0 | 462.0 |
| after_loss | 388.0 | 416.0 | 462.0 |
| after_backward | 1012.0 | 1030.0 | 1038.0 |
| after_optimizer_step | 1700.0 | 1738.0 | 1726.0 |
| after_zero_grad | 1700.0 | 1738.0 | 1726.0 |

## Key Stage-wise Finding

The highest peak memory occurred after optimizer_step.

For 128 tokens:

- after_backward peak allocated memory = 972.64 MB
- after_optimizer_step peak allocated memory = 1616.71 MB

This is the most important finding from Day 23.

It suggests that AdamW optimizer states are a major contributor to training memory.

AdamW usually stores extra optimizer state tensors such as:

- first moment
- second moment

These states become visible after the optimizer step.

## Training vs Inference Comparison

| input_tokens | training_peak_allocated_MB | training_peak_reserved_MB | inference_peak_allocated_MB | inference_peak_reserved_MB | allocated_training_vs_inference_ratio | reserved_training_vs_inference_ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1592.43 | 1700.0 | 332.18 | 360.0 | 4.793877 | 4.722222 |
| 64 | 1600.69 | 1738.0 | 335.36 | 364.0 | 4.773050 | 4.774725 |
| 64 | 1600.69 | 1738.0 | 335.36 | 364.0 | 4.773050 | 4.774725 |
| 128 | 1616.71 | 1726.0 | 343.24 | 366.0 | 4.710145 | 4.715847 |

## Training vs Inference Interpretation

For distilgpt2, training used around 4.7x to 4.8x more peak allocated memory than inference under comparable input lengths.

Examples:

- 32 tokens: training was 4.79x inference peak allocated memory
- 64 tokens: training was 4.77x inference peak allocated memory
- 128 tokens: training was 4.71x inference peak allocated memory

Reserved memory showed a similar ratio:

- 32 tokens: training was 4.72x inference peak reserved memory
- 64 tokens: training was 4.77x inference peak reserved memory
- 128 tokens: training was 4.72x inference peak reserved memory

This is a strong training-memory result.

It confirms that training memory is much larger than inference memory.

## Summary

| experiment | num_runs | min_input_tokens | max_input_tokens | min_peak_allocated_MB | max_peak_allocated_MB | min_peak_reserved_MB | max_peak_reserved_MB | oom_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2_training_sequence_length | 3 | 32 | 128 | 1592.43 | 1616.71 | 1700.0 | 1738.0 | 0 |

## Runtime Observation

The 32-token run took 9.323 seconds, while the 64-token and 128-token runs took around 1.5-1.6 seconds.

This should not be interpreted as 32 tokens being slower.

The first run likely included warmup overhead such as:

- model loading
- CUDA warmup
- optimizer initialization
- kernel startup
- Hugging Face loading/cache

For runtime analysis, repeated runs and warmup runs would be needed.

For this phase, memory is the main metric.

## Cleanup / Allocator Observation

For the 64-token and 128-token runs, before_model_load was not exactly zero:

- before_model_load allocated = 17.88 MB
- before_model_load reserved = 42 MB

This shows PyTorch/Colab allocator behavior.

Even after cleanup, PyTorch may retain some cached memory.

This is acceptable because before_model_load is recorded, but it should be mentioned clearly in the final report.

It also supports why allocator correction matters.

## Main Interpretation

Day 23 shows that distilgpt2 training memory is dominated by large fixed training components.

These include:

- model parameters
- gradients
- activations
- optimizer states
- allocator behavior

Sequence length does increase memory, but the increase from 32 to 128 tokens is small compared to the total training footprint.

The largest memory increase happened after optimizer_step, strongly suggesting that AdamW optimizer states are a major memory contributor.

This confirms that optimizer-state modeling is necessary for the training memory estimator.

## Why This Matters for the Project

This is one of the strongest project results so far.

It connects directly to the training-memory part of the original project goal.

The project now has evidence that:

- training memory is much larger than inference memory
- training memory scales with sequence length
- optimizer step can dominate peak memory
- AdamW optimizer states must be modeled
- allocator behavior affects reserved memory

## Questions Answered

### Q1. Did distilgpt2 training memory increase with sequence length?

Yes.

Peak allocated memory increased from 1592.43 MB at 32 tokens to 1616.71 MB at 128 tokens.

The increase was real but not huge because training memory is dominated by large fixed components like parameters, gradients, optimizer states, and overhead.

### Q2. Which sequence length used the highest peak memory?

For peak allocated memory:

128 tokens used the highest memory:

1616.71 MB

For peak reserved memory:

64 tokens had the highest reserved memory:

1738.0 MB

Reserved memory did not increase smoothly because PyTorch allocator reserves memory in chunks.

### Q3. Which stage created the highest peak memory?

The highest peak memory occurred after optimizer_step.

This was true across all tested sequence lengths.

For 128 tokens:

- after_backward peak allocated = 972.64 MB
- after_optimizer_step peak allocated = 1616.71 MB

This suggests that AdamW optimizer states are a major contributor to training memory.

### Q4. How much higher was training memory compared to inference memory?

Training memory was around 4.7x to 4.8x higher than inference memory for distilgpt2.

Examples:

- 32 tokens: 4.79x higher peak allocated memory
- 64 tokens: 4.77x higher peak allocated memory
- 128 tokens: 4.71x higher peak allocated memory

### Q5. Did any run OOM?

No.

All three distilgpt2 training runs completed successfully.

OOM count = 0

## Day 23 Conclusion

Day 23 completed the first meaningful distilgpt2 training sequence-length experiment.

Peak allocated memory increased from 1592.43 MB at 32 tokens to 1616.71 MB at 128 tokens.

Peak reserved memory stayed in the range of 1700 MB to 1738 MB.

The increase with sequence length was present but relatively small compared to the large fixed memory components in training.

The most important finding was that the highest peak memory occurred after optimizer_step.

For 128 tokens, peak allocated memory after backward was 972.64 MB, but after optimizer_step it reached 1616.71 MB.

This shows that AdamW optimizer states are a major contributor to training memory.

Compared to inference, distilgpt2 training used around 4.7x to 4.8x more peak allocated memory for similar input lengths.

No OOM occurred.

This confirms that training memory is dominated by parameters, gradients, optimizer states, activations, and allocator behavior.

The next step is Day 24: distilgpt2 training batch-size experiment.

---


# Day 24 - distilgpt2 Training Batch-size Experiment

## Goal

Measure how distilgpt2 training memory changes with batch size.

Day 23 measured sequence-length scaling.

Day 24 measured batch-size scaling.

This is important because batch size is one of the main factors that can cause training OOM.

## Files Created

- results/day24_distilgpt2_training_batch.csv
- results/day24_distilgpt2_training_batch_stages.csv
- results/day24_distilgpt2_batch_stage_peak_allocated_pivot.csv
- results/day24_distilgpt2_batch_stage_peak_reserved_pivot.csv
- results/day24_distilgpt2_training_batch_scaling.csv
- results/day24_distilgpt2_training_batch_vs_inference.csv
- results/day24_distilgpt2_training_batch_summary.csv
- report/day24_distilgpt2_training_batch.md
- plots/day24_distilgpt2_training_memory_vs_batch.png
- plots/day24_distilgpt2_training_memory_per_sample.png
- plots/day24_distilgpt2_allocated_by_stage.png
- plots/day24_distilgpt2_peak_allocated_by_stage.png

## Experiment Settings

- model = distilgpt2
- task = training
- input_tokens = 64
- dtype = fp32
- optimizer = AdamW
- learning_rate = 5e-5
- GPU = Tesla T4

Batch sizes tested:

- 1
- 2
- 4

## Main Results

| batch_size | input_tokens | loss | runtime_sec | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | oom |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 1.809547 | 9.727 | 1600.19 | 1738.0 | 972.10 | 1738.0 | False |
| 2 | 64 | 1.878980 | 1.362 | 1616.21 | 1726.0 | 987.25 | 1726.0 | False |
| 4 | 64 | 1.887785 | 1.362 | 1648.43 | 1768.0 | 1018.96 | 1768.0 | False |

## Main Memory Trend

Peak allocated memory increased with batch size:

- batch size 1: 1600.19 MB
- batch size 2: 1616.21 MB
- batch size 4: 1648.43 MB

From batch size 1 to batch size 4, peak allocated memory increased by:

48.24 MB

Peak reserved memory stayed in the range:

1726 MB to 1768 MB

Reserved memory did not increase smoothly because PyTorch CUDA allocator reserves memory in chunks.

## Batch Scaling Analysis

| batch_size | input_tokens | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | runtime_sec | oom | peak_allocated_per_sample_MB | peak_reserved_per_sample_MB |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 64 | 1600.19 | 1738.0 | 972.10 | 1738.0 | 9.727 | False | 1600.1900 | 1738.0 |
| 2 | 64 | 1616.21 | 1726.0 | 987.25 | 1726.0 | 1.362 | False | 808.1050 | 863.0 |
| 4 | 64 | 1648.43 | 1768.0 | 1018.96 | 1768.0 | 1.362 | False | 412.1075 | 442.0 |

## Per-sample Memory Interpretation

Peak allocated memory per sample decreased sharply as batch size increased:

- batch size 1: 1600.19 MB per sample
- batch size 2: 808.11 MB per sample
- batch size 4: 412.11 MB per sample

Peak reserved memory per sample also decreased:

- batch size 1: 1738.0 MB per sample
- batch size 2: 863.0 MB per sample
- batch size 4: 442.0 MB per sample

This shows that total memory increases with batch size, but memory per sample decreases.

Reason:

Large fixed costs such as model parameters, gradients, optimizer states, and allocator overhead are shared across the batch.

Only some components, especially activations and input tensors, scale with batch size.

## Stage Peak Allocated Pivot

| stage | batch 1 | batch 2 | batch 4 |
|---|---:|---:|---:|
| before_model_load | 0.00 | 17.88 | 17.88 |
| after_model_load | 313.23 | 331.73 | 331.73 |
| after_batch_creation | 313.24 | 331.74 | 331.74 |
| after_optimizer_creation | 313.24 | 331.74 | 331.74 |
| after_forward | 382.29 | 454.84 | 571.78 |
| after_loss | 382.29 | 454.84 | 571.78 |
| after_backward | 956.49 | 972.51 | 1002.98 |
| after_optimizer_step | 1600.19 | 1616.21 | 1648.43 |
| after_zero_grad | 1600.19 | 1616.21 | 1648.43 |

## Stage Peak Reserved Pivot

| stage | batch 1 | batch 2 | batch 4 |
|---|---:|---:|---:|
| before_model_load | 0.0 | 42.0 | 42.0 |
| after_model_load | 350.0 | 370.0 | 370.0 |
| after_batch_creation | 350.0 | 370.0 | 370.0 |
| after_optimizer_creation | 350.0 | 370.0 | 370.0 |
| after_forward | 416.0 | 462.0 | 616.0 |
| after_loss | 416.0 | 462.0 | 616.0 |
| after_backward | 1030.0 | 1038.0 | 1160.0 |
| after_optimizer_step | 1738.0 | 1726.0 | 1768.0 |
| after_zero_grad | 1738.0 | 1726.0 | 1768.0 |

## Key Stage-wise Finding

The highest peak memory again occurred after optimizer_step.

For batch size 4:

- after_backward peak allocated memory = 1002.98 MB
- after_optimizer_step peak allocated memory = 1648.43 MB

This is the same pattern seen on Day 23.

This strengthens the conclusion that AdamW optimizer states are a major contributor to training memory.

## Training vs Inference Batch Comparison

The inference comparison table contained multiple repeated batch-size-1 rows because the inference dataset has many repeated distilgpt2 experiments at input_tokens=64.

So this table should be deduplicated before final reporting.

Still, the useful comparison is:

| batch_size | training_peak_allocated_MB | inference_peak_allocated_MB | allocated_training_vs_inference_ratio |
|---:|---:|---:|---:|
| 1 | 1600.19 | around 335-341 MB | around 4.7x |
| 2 | 1616.21 | 343.24 MB | 4.71x |
| 4 | 1648.43 | 354.49 MB | 4.65x |

The main takeaway is that distilgpt2 training uses around 4.6x to 4.8x more peak allocated memory than inference under comparable settings.

## Summary

| experiment | num_runs | min_batch_size | max_batch_size | min_peak_allocated_MB | max_peak_allocated_MB | min_peak_reserved_MB | max_peak_reserved_MB | oom_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2_training_batch_size | 3 | 1 | 4 | 1600.19 | 1648.43 | 1726.0 | 1768.0 | 0 |

## Runtime Observation

Batch size 1 took 9.727 seconds, while batch sizes 2 and 4 took around 1.362 seconds.

This should not be interpreted as batch size 1 being slower.

The first run likely included warmup overhead such as:

- model loading
- CUDA warmup
- optimizer initialization
- kernel startup
- Hugging Face loading/cache

For runtime analysis, repeated runs and warmup runs would be needed.

For this phase, memory is the main metric.

## Batch-size Scaling Interpretation

Batch-size scaling was strongly sublinear.

Batch size increased from 1 to 4, but peak allocated memory only increased from 1600.19 MB to 1648.43 MB.

If total memory scaled linearly, batch size 4 would have used around 4 times the memory of batch size 1.

That did not happen because most training memory is dominated by shared fixed costs:

- model parameters
- gradients
- AdamW optimizer states
- allocator overhead

Batch-dependent memory, mainly activations, increased with batch size, but it was only part of total training memory.

## Main Interpretation

Day 24 showed that batch size increases training memory, but not linearly.

Peak allocated memory increased by 48.24 MB from batch size 1 to batch size 4.

Memory per sample decreased sharply because fixed model and optimizer costs were shared across the batch.

The highest peak memory again occurred after optimizer_step, confirming the Day 23 observation that AdamW optimizer states are a major contributor to training memory.

No OOM occurred for batch sizes 1, 2, and 4.

## Why This Matters for the Project

This experiment supports the training-memory estimator design.

A training memory estimator should include:

- fixed model parameter memory
- gradient memory
- optimizer state memory
- activation memory that depends on batch size and sequence length
- allocator correction

The result shows that batch size affects training memory, but fixed memory components dominate for this model and setup.

## Questions Answered

### Q1. Did training memory increase with batch size?

Yes.

Peak allocated memory increased from 1600.19 MB at batch size 1 to 1648.43 MB at batch size 4.

Peak reserved memory stayed in the range of 1726 MB to 1768 MB.

### Q2. Was batch-size scaling linear or sublinear?

Batch-size scaling was strongly sublinear.

Batch size increased 4x, but peak allocated memory only increased by 48.24 MB.

This happened because model parameters, gradients, and optimizer states are shared across the batch.

### Q3. Which batch size used the highest peak memory?

Batch size 4 used the highest peak memory.

- peak allocated memory = 1648.43 MB
- peak reserved memory = 1768 MB

### Q4. Which stage caused the highest peak memory?

The highest peak memory occurred after optimizer_step.

For batch size 4:

- after_backward peak allocated = 1002.98 MB
- after_optimizer_step peak allocated = 1648.43 MB

This supports the optimizer-state memory hypothesis.

### Q5. Did any run OOM?

No.

All three batch sizes completed successfully.

OOM count = 0

## Day 24 Conclusion

Day 24 completed the distilgpt2 training batch-size experiment.

Peak allocated memory increased from 1600.19 MB at batch size 1 to 1648.43 MB at batch size 4.

Peak reserved memory stayed between 1726 MB and 1768 MB.

The increase with batch size was real but strongly sublinear because most memory was dominated by fixed components such as model parameters, gradients, AdamW optimizer states, and allocator overhead.

Memory per sample decreased sharply from 1600.19 MB at batch size 1 to 412.11 MB at batch size 4, showing that fixed model/optimizer costs are amortized across the batch.

The highest peak memory again occurred after optimizer_step.

For batch size 4, peak allocated memory after backward was 1002.98 MB, while after optimizer_step it reached 1648.43 MB.

This further confirms that AdamW optimizer states are a major contributor to training memory.

Compared to inference, distilgpt2 training used around 4.6x to 4.8x more peak allocated memory under comparable settings.

No OOM occurred.

The next step is Day 25: distilgpt2 optimizer comparison between SGD and AdamW.

---


# Day 25 - distilgpt2 Optimizer Comparison

## Goal

Compare training memory between SGD and AdamW for distilgpt2.

Day 23 and Day 24 showed that the highest training memory peak happened after optimizer_step.

Day 25 isolated optimizer choice as the changed variable.

This experiment is important because it gives direct evidence for building the OptimizerStateEstimator.

## Files Created

- results/day25_distilgpt2_optimizer_comparison.csv
- results/day25_distilgpt2_optimizer_comparison_stages.csv
- results/day25_optimizer_stage_peak_allocated_pivot.csv
- results/day25_optimizer_stage_peak_reserved_pivot.csv
- results/day25_optimizer_stage_current_allocated_pivot.csv
- results/day25_optimizer_stage_current_reserved_pivot.csv
- results/day25_optimizer_memory_difference.csv
- results/day25_optimizer_step_stage_comparison.csv
- results/day25_optimizer_step_jump.csv
- results/day25_optimizer_comparison_summary.csv
- report/day25_distilgpt2_optimizer_comparison.md
- plots/day25_optimizer_peak_allocated.png
- plots/day25_optimizer_peak_reserved.png
- plots/day25_optimizer_allocated_by_stage.png
- plots/day25_optimizer_peak_allocated_by_stage.png

## Experiment Settings

- model = distilgpt2
- task = training
- batch_size = 1
- input_tokens = 64
- dtype = fp32
- learning_rate = 5e-5
- GPU = Tesla T4

Optimizers tested:

- SGD
- AdamW

## Main Results

| optimizer_name | loss | runtime_sec | peak_allocated_MB | peak_reserved_MB | final_allocated_MB | final_reserved_MB | oom |
|---|---:|---:|---:|---:|---:|---:|---|
| sgd | 1.804632 | 9.183 | 956.49 | 1030.0 | 345.63 | 1030.0 | False |
| adamw | 1.836824 | 1.441 | 1600.94 | 1738.0 | 973.48 | 1738.0 | False |

## Main Result Interpretation

AdamW used much more memory than SGD.

Peak allocated memory:

- SGD = 956.49 MB
- AdamW = 1600.94 MB

Peak reserved memory:

- SGD = 1030.0 MB
- AdamW = 1738.0 MB

Final allocated memory:

- SGD = 345.63 MB
- AdamW = 973.48 MB

Final reserved memory:

- SGD = 1030.0 MB
- AdamW = 1738.0 MB

This is a major training-memory result.

## Optimizer Memory Difference

| metric | sgd | adamw | difference_MB | percent_increase |
|---|---:|---:|---:|---:|
| peak_allocated_MB | 956.49 | 1600.94 | 644.45 | 67.376554 |
| peak_reserved_MB | 1030.00 | 1738.00 | 708.00 | 68.737864 |
| final_allocated_MB | 345.63 | 973.48 | 627.85 | 181.653792 |
| final_reserved_MB | 1030.00 | 1738.00 | 708.00 | 68.737864 |

## Main Optimizer Difference

AdamW increased peak allocated memory by:

644.45 MB

This is a 67.38% increase compared to SGD.

AdamW increased peak reserved memory by:

708 MB

This is a 68.74% increase compared to SGD.

AdamW increased final allocated memory by:

627.85 MB

This is a 181.65% increase compared to SGD.

## Peak Allocated Pivot

| stage | adamw | sgd |
|---|---:|---:|
| before_model_load | 17.88 | 0.00 |
| after_model_load | 331.73 | 313.23 |
| after_batch_creation | 331.74 | 313.24 |
| after_optimizer_creation | 331.74 | 313.24 |
| after_forward | 391.04 | 382.29 |
| after_loss | 391.04 | 382.29 |
| after_backward | 956.49 | 956.49 |
| after_optimizer_step | 1600.94 | 956.49 |
| after_zero_grad | 1600.94 | 956.49 |

## Peak Reserved Pivot

| stage | adamw | sgd |
|---|---:|---:|
| before_model_load | 42.0 | 0.0 |
| after_model_load | 370.0 | 350.0 |
| after_batch_creation | 370.0 | 350.0 |
| after_optimizer_creation | 370.0 | 350.0 |
| after_forward | 436.0 | 416.0 |
| after_loss | 436.0 | 416.0 |
| after_backward | 1050.0 | 1030.0 |
| after_optimizer_step | 1738.0 | 1030.0 |
| after_zero_grad | 1738.0 | 1030.0 |

## Current Allocated Pivot

| stage | adamw | sgd |
|---|---:|---:|
| before_model_load | 17.88 | 0.00 |
| after_model_load | 331.73 | 313.23 |
| after_batch_creation | 331.74 | 313.24 |
| after_optimizer_creation | 331.74 | 313.24 |
| after_forward | 391.04 | 382.29 |
| after_loss | 391.04 | 382.29 |
| after_backward | 660.49 | 660.49 |
| after_optimizer_step | 1287.71 | 660.49 |
| after_zero_grad | 973.48 | 345.63 |

## Current Reserved Pivot

| stage | adamw | sgd |
|---|---:|---:|
| before_model_load | 42.0 | 0.0 |
| after_model_load | 370.0 | 350.0 |
| after_batch_creation | 370.0 | 350.0 |
| after_optimizer_creation | 370.0 | 350.0 |
| after_forward | 436.0 | 416.0 |
| after_loss | 436.0 | 416.0 |
| after_backward | 1050.0 | 1030.0 |
| after_optimizer_step | 1738.0 | 1030.0 |
| after_zero_grad | 1738.0 | 1030.0 |

## Important Stage-wise Finding

After backward, both optimizers had the same peak allocated memory:

- SGD after_backward peak allocated = 956.49 MB
- AdamW after_backward peak allocated = 956.49 MB

This shows that backward memory was the same for both optimizers.

The difference appeared during optimizer_step.

For SGD:

- after_backward peak allocated = 956.49 MB
- after_optimizer_step peak allocated = 956.49 MB

So SGD had no peak memory jump during optimizer_step.

For AdamW:

- after_backward peak allocated = 956.49 MB
- after_optimizer_step peak allocated = 1600.94 MB

So AdamW had a large peak memory jump during optimizer_step.

## Optimizer Step Stage Comparison

| optimizer_name | stage | allocated_MB | reserved_MB | peak_allocated_MB | peak_reserved_MB |
|---|---|---:|---:|---:|---:|
| sgd | after_backward | 660.49 | 1030.0 | 956.49 | 1030.0 |
| sgd | after_optimizer_step | 660.49 | 1030.0 | 956.49 | 1030.0 |
| sgd | after_zero_grad | 345.63 | 1030.0 | 956.49 | 1030.0 |
| adamw | after_backward | 660.49 | 1050.0 | 956.49 | 1050.0 |
| adamw | after_optimizer_step | 1287.71 | 1738.0 | 1600.94 | 1738.0 |
| adamw | after_zero_grad | 973.48 | 1738.0 | 1600.94 | 1738.0 |

## Optimizer Step Jump

| optimizer_name | allocated_jump_backward_to_step_MB | reserved_jump_backward_to_step_MB | peak_allocated_jump_backward_to_step_MB | peak_reserved_jump_backward_to_step_MB | allocated_after_zero_grad_MB | reserved_after_zero_grad_MB |
|---|---:|---:|---:|---:|---:|---:|
| sgd | 0.00 | 0.0 | 0.00 | 0.0 | 345.63 | 1030.0 |
| adamw | 627.22 | 688.0 | 644.45 | 688.0 | 973.48 | 1738.0 |

## Main Optimizer Step Finding

For SGD:

- allocated jump from backward to optimizer_step = 0.00 MB
- reserved jump from backward to optimizer_step = 0.0 MB
- peak allocated jump from backward to optimizer_step = 0.00 MB

For AdamW:

- allocated jump from backward to optimizer_step = 627.22 MB
- reserved jump from backward to optimizer_step = 688.0 MB
- peak allocated jump from backward to optimizer_step = 644.45 MB

This is the clearest evidence from Day 25.

It shows that AdamW optimizer states are created or updated during optimizer_step and contribute heavily to training memory.

## Why AdamW Uses More Memory

SGD without momentum stores very little persistent optimizer state.

AdamW stores extra optimizer state tensors for each parameter.

These usually include:

- first moment estimate
- second moment estimate

This means AdamW requires much more memory than plain SGD.

The experiment shows this clearly because both optimizers had the same memory after backward, but AdamW jumped sharply after optimizer_step.

## Small Cleanup Warning

The AdamW run started with some cached memory:

- before_model_load allocated = 17.88 MB
- before_model_load reserved = 42.0 MB

The SGD run started clean:

- before_model_load allocated = 0.00 MB
- before_model_load reserved = 0.0 MB

This creates a small baseline difference.

However, this does not explain the 644.45 MB peak allocated difference.

The AdamW optimizer-state effect is much larger than the small cached-memory baseline.

So the conclusion is still valid.

## Summary

| experiment | num_runs | optimizers_tested | min_peak_allocated_MB | max_peak_allocated_MB | min_peak_reserved_MB | max_peak_reserved_MB | oom_count |
|---|---:|---|---:|---:|---:|---:|---:|
| distilgpt2_optimizer_comparison | 2 | sgd, adamw | 956.49 | 1600.94 | 1030.0 | 1738.0 | 0 |

## Main Interpretation

Day 25 directly proves that optimizer choice has a major effect on training memory.

SGD and AdamW had the same backward peak memory, but AdamW used much more memory after optimizer_step.

This means optimizer-state memory is separate from backward/gradient memory.

The result strongly supports adding an OptimizerStateEstimator to the project.

## Why This Matters for the Project

The training estimator should include:

- parameter memory
- gradient memory
- activation memory
- optimizer state memory
- allocator overhead

Day 25 gives direct evidence for the optimizer-state part.

For AdamW, optimizer state memory should be modeled as roughly two extra parameter-sized tensors.

For SGD without momentum, optimizer state memory should be close to zero.

## Questions Answered

### Q1. Which optimizer used more peak allocated memory?

AdamW used more peak allocated memory.

- SGD peak allocated = 956.49 MB
- AdamW peak allocated = 1600.94 MB

AdamW used 644.45 MB more peak allocated memory.

### Q2. Which optimizer used more final allocated memory?

AdamW used more final allocated memory.

- SGD final allocated = 345.63 MB
- AdamW final allocated = 973.48 MB

AdamW used 627.85 MB more final allocated memory.

### Q3. What changed after optimizer_step?

For SGD, optimizer_step added no extra peak memory.

For AdamW, optimizer_step caused a large memory jump:

- allocated memory increased by 627.22 MB
- reserved memory increased by 688.0 MB
- peak allocated memory increased by 644.45 MB
- peak reserved memory increased by 688.0 MB

This means AdamW optimizer state tensors became visible during optimizer_step.

### Q4. Why does AdamW need more memory than SGD?

AdamW stores optimizer states such as first moment and second moment tensors for each parameter.

SGD without momentum stores little or no extra persistent optimizer state.

Therefore, AdamW uses much more memory than SGD.

### Q5. How does this support OptimizerStateEstimator?

This experiment shows that optimizer memory must be modeled separately.

A training memory estimator should include an optimizer-state component.

For AdamW, the optimizer-state component should be around two extra parameter-sized tensors.

For plain SGD, the optimizer-state component should be near zero.

## Day 25 Conclusion

Day 25 completed the distilgpt2 optimizer comparison between SGD and AdamW.

The result clearly showed optimizer-state memory impact.

SGD reached 956.49 MB peak allocated memory and 1030 MB peak reserved memory.

AdamW reached 1600.94 MB peak allocated memory and 1738 MB peak reserved memory.

AdamW increased peak allocated memory by 644.45 MB, or 67.38%, compared to SGD.

It also increased peak reserved memory by 708 MB, or 68.74%.

The most important finding came from stage-wise analysis.

After backward, both SGD and AdamW had the same peak allocated memory of 956.49 MB.

During optimizer_step, SGD showed no memory increase, while AdamW increased current allocated memory by 627.22 MB and peak allocated memory by 644.45 MB.

This shows that AdamW optimizer states are a major contributor to training memory.

This result strongly supports building an OptimizerStateEstimator, where SGD has little extra optimizer state and AdamW adds approximately two extra parameter-sized state tensors.

The next step is Day 26: build OptimizerStateEstimator.

---


# Day 26 - OptimizerStateEstimator

## Goal

Build an optimizer-state memory estimator based on the Day 25 SGD vs AdamW experiment.

Day 25 showed that AdamW caused a large memory increase after optimizer_step, while SGD did not.

Day 26 converted this observation into a reusable estimator module.

## Files Created

- src/estimators/optimizer_state_estimator.py
- results/day26_optimizer_state_predictions.csv
- results/day26_optimizer_state_difference_summary.csv
- results/day26_optimizer_state_jump_comparison.csv
- results/day26_optimizer_state_metrics.csv
- report/day26_optimizer_state_estimator.md
- plots/day26_optimizer_state_estimate_vs_observed.png

## Estimator Rule

The OptimizerStateEstimator uses optimizer-specific memory factors:

- SGD: 0 x parameter memory
- SGD with momentum: 1 x parameter memory
- Adam: 2 x parameter memory
- AdamW: 2 x parameter memory

For this experiment, the main comparison was SGD vs AdamW.

## Why This Rule Makes Sense

Plain SGD without momentum stores little or no persistent optimizer state.

AdamW stores extra tensors for each parameter, mainly:

- first moment estimate
- second moment estimate

So AdamW optimizer-state memory is approximated as:

2 x parameter memory

## Optimizer State Predictions

| model_name | optimizer_name | dtype | actual_peak_allocated_MB | actual_peak_reserved_MB | final_allocated_MB | final_reserved_MB | parameter_memory_MB | optimizer_state_factor | estimated_optimizer_state_MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | sgd | fp32 | 956.49 | 1030.0 | 345.63 | 1030.0 | 312.47168 | 0.0 | 0.000000 |
| distilgpt2 | adamw | fp32 | 1600.94 | 1738.0 | 973.48 | 1738.0 | 312.47168 | 2.0 | 624.943359 |

## Main Prediction Result

For distilgpt2 in fp32:

- parameter memory = 312.47 MB
- SGD estimated optimizer-state memory = 0.00 MB
- AdamW estimated optimizer-state memory = 624.94 MB

The AdamW estimate comes from:

2 x 312.47 MB = 624.94 MB

## Difference Summary

| model_name | observed_peak_allocated_difference_MB | observed_peak_reserved_difference_MB | estimated_optimizer_state_difference_MB | absolute_error_vs_peak_allocated_difference_MB | relative_error_vs_peak_allocated_difference_percent |
|---|---:|---:|---:|---:|---:|
| distilgpt2 | 644.45 | 708.0 | 624.943359 | 19.506641 | 3.026866 |

## Difference Interpretation

The observed AdamW-SGD peak allocated memory difference was:

644.45 MB

The estimated optimizer-state memory difference was:

624.94 MB

Absolute error:

19.51 MB

Relative error:

3.03%

This is a strong result.

It shows that the simple AdamW rule of 2 x parameter memory explains most of the observed memory gap.

## Optimizer Step Jump Comparison

| model_name | optimizer_name | estimated_optimizer_state_MB | actual_allocated_jump_backward_to_step_MB | actual_peak_allocated_jump_backward_to_step_MB | actual_reserved_jump_backward_to_step_MB | error_vs_current_allocated_jump_MB | error_vs_peak_allocated_jump_MB |
|---|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | adamw | 624.943359 | 627.22 | 644.45 | 688.0 | 2.276641 | 19.506641 |

## Optimizer Step Jump Interpretation

The estimator matched the actual AdamW optimizer-step jump very closely.

AdamW current allocated memory jump from after_backward to after_optimizer_step:

627.22 MB

Estimated optimizer-state memory:

624.94 MB

Error:

2.28 MB

This is the cleanest evidence that the estimator is modeling the real optimizer-state memory.

Compared to peak allocated jump:

- actual peak jump = 644.45 MB
- estimated = 624.94 MB
- error = 19.51 MB

Both comparisons are strong.

## Metrics

| model_name | optimizer | estimated_optimizer_state_MB | observed_peak_allocated_difference_MB | observed_peak_reserved_difference_MB | absolute_error_MB | relative_error_percent |
|---|---|---:|---:|---:|---:|---:|
| distilgpt2 | adamw | 624.943359 | 644.45 | 708.0 | 19.506641 | 3.026866 |

## Main Interpretation

Day 26 successfully built the OptimizerStateEstimator.

The estimator models optimizer-state memory as a function of model parameter memory and optimizer type.

For AdamW, optimizer state is estimated as 2 x parameter memory.

For SGD without momentum, optimizer-state memory is estimated as 0.

The Day 25 experiment supports this strongly because AdamW had a large memory jump after optimizer_step, while SGD did not.

## Why This Matters for the Project

Training memory cannot be estimated correctly using only parameters and activations.

Optimizer state is a major part of training memory.

Day 25 showed that AdamW added around 644 MB peak allocated memory compared to SGD.

Day 26 showed that this can be predicted using the simple rule:

AdamW optimizer state ≈ 2 x parameter memory

This estimator will become one component of the full TrainingMemoryEstimator.

## Questions Answered

### Q1. Why does AdamW have optimizer-state memory?

AdamW stores extra state tensors for each parameter.

These mainly include first moment and second moment estimates.

These tensors are needed for adaptive optimization.

### Q2. Why is SGD modeled with zero optimizer-state memory?

Plain SGD without momentum does not store persistent optimizer state like AdamW.

It mostly applies gradients directly to parameters.

So extra optimizer-state memory is close to zero.

### Q3. What optimizer-state factor did you use for AdamW?

AdamW uses an optimizer-state factor of:

2 x parameter memory

This is because AdamW stores approximately two parameter-sized state tensors.

### Q4. How close was the estimated optimizer-state memory to the observed AdamW-SGD peak difference?

The estimate was very close.

- estimated optimizer state = 624.94 MB
- observed AdamW-SGD peak allocated difference = 644.45 MB
- absolute error = 19.51 MB
- relative error = 3.03%

Against the current allocated optimizer-step jump:

- actual jump = 627.22 MB
- estimated = 624.94 MB
- error = 2.28 MB

### Q5. How will this be used in TrainingMemoryEstimator?

It will become the optimizer-state component of the training memory estimator.

The full training estimator will include:

- parameter memory
- gradient memory
- activation memory
- optimizer-state memory
- overhead / allocator correction

For AdamW:

optimizer_state_memory ≈ 2 x parameter memory

For SGD:

optimizer_state_memory ≈ 0

## Day 26 Conclusion

Day 26 built and evaluated the OptimizerStateEstimator.

For distilgpt2 in fp32, the estimator calculated parameter memory as 312.47 MB.

Since AdamW stores two optimizer-state tensors per parameter, the estimated AdamW optimizer-state memory was 624.94 MB.

The observed AdamW-SGD peak allocated memory difference was 644.45 MB.

The estimator had only 19.51 MB absolute error, corresponding to 3.03% relative error.

The estimator was even closer to the current allocated optimizer-step jump.

AdamW increased current allocated memory by 627.22 MB from after_backward to after_optimizer_step, while the estimator predicted 624.94 MB, giving only 2.28 MB error.

This confirms that modeling AdamW optimizer state as approximately 2 x parameter memory is a strong approximation for this experiment.

This module can now be used inside the TrainingMemoryEstimator.

The next step is Day 27: build TrainingMemoryEstimator.

---


# Day 27 - TrainingMemoryEstimator

## Goal

Build the first full training memory estimator.

Day 26 created the OptimizerStateEstimator.

Day 27 used it inside a full TrainingMemoryEstimator.

The goal was to create an explainable baseline estimator for training peak memory, not to make a perfect final model.

## Files Created

- src/estimators/training_memory_estimator.py
- results/day27_training_estimator_predictions.csv
- results/day27_training_estimator_metrics.csv
- results/day27_training_error_by_optimizer.csv
- results/day27_training_error_by_experiment.csv
- results/day27_training_component_summary.csv
- report/day27_training_memory_estimator.md
- plots/day27_training_actual_vs_predicted_allocated.png
- plots/day27_training_actual_vs_predicted_reserved.png
- plots/day27_training_components_sgd.png
- plots/day27_training_components_adamw.png

## Estimator Formula

The first version of the TrainingMemoryEstimator uses:

predicted_peak_allocated =
parameter_memory
+ gradient_memory
+ optimizer_state_memory
+ activation_memory
+ framework_overhead

predicted_peak_reserved =
predicted_peak_allocated
+ allocator_padding

## Estimator Components

The estimator includes:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- framework overhead
- allocator padding

## Estimator Settings

- activation_factor = 0.004
- framework_overhead_mb = 20.0
- allocator_padding_ratio = 0.08
- allocator_min_padding_mb = 32.0

## Global Metrics

| num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 15.655726 | 14.448069 | 24.208710 | 3.553216 | 16.117962 | 14.133240 | 23.987282 | 2.881995 |

## Main Global Result

The first TrainingMemoryEstimator achieved:

- allocated MRE = 15.66%
- allocated mean error = 14.45%
- reserved MRE = 16.12%
- reserved mean error = 14.13%

This is a decent first baseline.

The estimator is not final yet.

It systematically underpredicts training memory, which means a training/backward overhead correction is still missing.

## Error by Optimizer

| optimizer_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 7 | 13.053691 | 15.642551 | 17.743129 | 12.725520 | 16.117962 | 16.783898 | 1610.80 | 1401.303379 | 1733.428571 | 1513.407650 |
| sgd | 1 | 24.208710 | 24.208710 | 24.208710 | 23.987282 | 23.987282 | 23.987282 | 956.49 | 724.936109 | 1030.000000 | 782.930998 |

## Optimizer-wise Interpretation

The estimator performs better on AdamW rows than on the SGD row.

For AdamW:

- allocated mean error = 13.05%
- reserved mean error = 12.73%

For SGD:

- allocated error = 24.21%
- reserved error = 23.99%

The AdamW separation is logically correct because AdamW includes optimizer-state memory.

The SGD row is worse because the current estimator does not include enough training/backward temporary overhead.

## Error by Experiment Type

| experiment_type | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2_batch_size_variation | 3 | 10.241691 | 11.529305 | 15.642551 | 9.843130 | 10.529432 | 16.117962 | 1621.610000 | 1456.536469 | 1744.000000 | 1573.059386 |
| distilgpt2_optimizer_comparison | 2 | 19.945390 | 19.945390 | 24.208710 | 20.052622 | 20.052622 | 23.987282 | 1278.715000 | 1037.407789 | 1384.000000 | 1120.400412 |
| distilgpt2_sequence_length_variation | 3 | 14.989565 | 15.668901 | 17.743129 | 14.477097 | 16.117962 | 16.783898 | 1603.276667 | 1363.211594 | 1721.333333 | 1472.268521 |

## Experiment-wise Interpretation

The estimator performed best on batch-size variation:

- allocated mean error = 10.24%
- reserved mean error = 9.84%

It performed worst on optimizer comparison:

- allocated mean error = 19.95%
- reserved mean error = 20.05%

This happens because the SGD row is underpredicted more heavily than AdamW rows.

## Prediction Results

| model_name | batch_size | input_tokens | optimizer_name | experiment_type | actual_peak_allocated_MB | predicted_peak_allocated_MB | allocated_relative_error_percent | actual_peak_reserved_MB | predicted_peak_reserved_MB | reserved_relative_error_percent | parameter_memory_MB | gradient_memory_MB | optimizer_state_memory_MB | activation_memory_MB |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 1 | 32 | adamw | distilgpt2_sequence_length_variation | 1592.43 | 1309.883094 | 17.743129 | 1700.0 | 1414.673741 | 16.783898 | 312.47168 | 312.47168 | 624.943359 | 39.996375 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_sequence_length_variation | 1600.69 | 1349.879469 | 15.668901 | 1738.0 | 1457.869826 | 16.117962 | 312.47168 | 312.47168 | 624.943359 | 79.992750 |
| distilgpt2 | 1 | 128 | adamw | distilgpt2_sequence_length_variation | 1616.71 | 1429.872219 | 11.556666 | 1726.0 | 1544.261996 | 10.529432 | 312.47168 | 312.47168 | 624.943359 | 159.985500 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_batch_size_variation | 1600.19 | 1349.879469 | 15.642551 | 1738.0 | 1457.869826 | 16.117962 | 312.47168 | 312.47168 | 624.943359 | 79.992750 |
| distilgpt2 | 2 | 64 | adamw | distilgpt2_batch_size_variation | 1616.21 | 1429.872219 | 11.529305 | 1726.0 | 1544.261996 | 10.529432 | 312.47168 | 312.47168 | 624.943359 | 159.985500 |
| distilgpt2 | 4 | 64 | adamw | distilgpt2_batch_size_variation | 1648.43 | 1589.857719 | 3.553216 | 1768.0 | 1717.046336 | 2.881995 | 312.47168 | 312.47168 | 624.943359 | 319.971000 |
| distilgpt2 | 1 | 64 | sgd | distilgpt2_optimizer_comparison | 956.49 | 724.936109 | 24.208710 | 1030.0 | 782.930998 | 23.987282 | 312.47168 | 312.47168 | 0.000000 | 79.992750 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_optimizer_comparison | 1600.94 | 1349.879469 | 15.682070 | 1738.0 | 1457.869826 | 16.117962 | 312.47168 | 312.47168 | 624.943359 | 79.992750 |

## Best Prediction

The best prediction was:

- optimizer = AdamW
- batch_size = 4
- input_tokens = 64

Actual peak allocated memory:

1648.43 MB

Predicted peak allocated memory:

1589.86 MB

Allocated relative error:

3.55%

Reserved relative error:

2.88%

This was the best row because activation memory is larger at batch size 4, so the activation component helped reduce underprediction.

## Worst Prediction

The worst prediction was the SGD row:

Actual peak allocated memory:

956.49 MB

Predicted peak allocated memory:

724.94 MB

Allocated relative error:

24.21%

Reserved relative error:

23.99%

This shows that the estimator is missing training/backward temporary overhead beyond parameters, gradients, and activations.

## Component Summary

| optimizer_name | parameter_memory_MB | gradient_memory_MB | optimizer_state_memory_MB | activation_memory_MB | framework_overhead_MB |
|---|---:|---:|---:|---:|---:|
| adamw | 312.47168 | 312.47168 | 624.943359 | 131.416661 | 20.0 |
| sgd | 312.47168 | 312.47168 | 0.000000 | 79.992750 | 20.0 |

## Component Interpretation

For AdamW rows, the main components are:

- parameter memory = 312.47 MB
- gradient memory = 312.47 MB
- optimizer-state memory = 624.94 MB
- activation memory varies with batch size and sequence length

For SGD, optimizer-state memory is zero.

This is why AdamW prediction is much higher than SGD prediction.

The component structure is logically correct.

The problem is that both SGD and AdamW still need an additional training/backward temporary overhead term.

## Main Interpretation

Day 27 created the first full TrainingMemoryEstimator.

It combines parameter memory, gradient memory, optimizer-state memory, activation memory, framework overhead, and allocator padding.

The estimator is explainable and modular.

The first version achieved:

- allocated MRE = 15.66%
- reserved MRE = 16.12%

The estimator is systematically underpredicting training memory.

This means the optimizer-state component works, but the model needs a better correction for:

- backward temporary memory
- framework overhead
- training runtime overhead
- allocator behavior

## Why This Matters for the Project

Before Day 27, the project had training profiling results but no full training estimator.

After Day 27, the project has a complete baseline training-memory prediction pipeline.

This turns the training phase from simple measurement into memory prediction.

## Questions Answered

### Q1. What components are included in the TrainingMemoryEstimator?

The TrainingMemoryEstimator includes:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- framework overhead
- allocator padding

### Q2. Why do we include gradient memory separately from parameter memory?

During training, both parameters and gradients exist in memory.

Gradients are usually parameter-sized tensors.

So parameter memory and gradient memory must be counted separately.

### Q3. Why does AdamW prediction become higher than SGD prediction?

AdamW prediction becomes higher because AdamW includes optimizer-state memory.

AdamW optimizer state is modeled as:

2 x parameter memory

SGD without momentum has approximately zero extra optimizer-state memory.

### Q4. What does activation memory depend on?

In the estimator, activation memory depends on:

- parameter memory
- batch size
- input tokens
- dtype
- activation_factor

Practically, activation memory grows with batch size and sequence length.

### Q5. Which part of the estimator likely needs tuning next?

The missing part is training/backward temporary overhead.

The estimator underpredicts memory, especially for SGD.

So the next improvement should add a backward temporary memory correction or increase the training overhead term.

## Day 27 Conclusion

Day 27 built the first full TrainingMemoryEstimator.

The estimator was evaluated on 8 distilgpt2 training rows from sequence-length, batch-size, and optimizer-comparison experiments.

The first version achieved 15.66% allocated MRE and 16.12% reserved MRE.

Mean allocated error was 14.45%, and mean reserved error was 14.13%.

The estimator correctly separated AdamW from SGD because AdamW includes optimizer-state memory while SGD does not.

For AdamW rows, allocated mean error was 13.05%.

For the SGD row, allocated error was 24.21%.

The estimator systematically underpredicted training memory.

This shows that the optimizer-state component works, but the estimator is still missing enough backward temporary memory, framework overhead, or training runtime overhead.

The next step is Day 28: improve the estimator by adding a training overhead or backward temporary memory correction term.

---


# Day 28 - TrainingMemoryEstimator Improvement

## Goal

Improve the first TrainingMemoryEstimator.

Day 27 created TrainingMemoryEstimator V1, but it systematically underpredicted training memory.

Day 28 improved the estimator by adding a backward temporary memory correction term.

## Files Created

- results/day28_training_estimator_v2_predictions.csv
- results/day28_training_estimator_v2_metrics.csv
- results/day28_training_estimator_v1_vs_v2_metrics.csv
- results/day28_training_estimator_v1_vs_v2_row_comparison.csv
- results/day28_training_v2_error_by_optimizer.csv
- results/day28_training_v2_error_by_experiment.csv
- results/day28_training_v2_component_summary.csv
- report/day28_training_estimator_improvement.md
- plots/day28_v1_vs_v2_allocated_mre.png
- plots/day28_v1_vs_v2_reserved_mre.png
- plots/day28_v2_actual_vs_predicted_allocated.png

## Main Formula Change

V1 formula:

predicted_peak_allocated =
parameter_memory
+ gradient_memory
+ optimizer_state_memory
+ activation_memory
+ framework_overhead

V2 formula:

predicted_peak_allocated =
parameter_memory
+ gradient_memory
+ optimizer_state_memory
+ activation_memory
+ backward_temp_memory
+ framework_overhead

The new component is:

backward_temp_memory_MB = parameter_memory_MB x backward_temp_factor

where:

backward_temp_factor = 0.65

For distilgpt2:

- parameter memory = 312.47 MB
- backward temporary memory = 203.11 MB

## Why This Change Was Needed

Training creates temporary tensors during backward execution.

V1 did not include this backward temporary memory.

Because of that, V1 underpredicted memory, especially for the SGD row.

Adding backward_temp_memory fixed most of the systematic underprediction.

## V1 vs V2 Metrics

| estimator | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainingMemoryEstimatorV1 | 8 | 15.655726 | 14.448069 | 24.208710 | 3.553216 | 16.117962 | 14.133240 | 23.987282 | 2.881995 |
| TrainingMemoryEstimatorV2 | 8 | 2.977173 | 3.462507 | 8.767998 | 1.006291 | 3.496839 | 3.868211 | 9.524969 | 2.179439 |

## Main Improvement

Allocated MRE improved from:

15.66% to 2.98%

Reserved MRE improved from:

16.12% to 3.50%

Mean allocated error improved from:

14.45% to 3.46%

Mean reserved error improved from:

14.13% to 3.87%

This is a major improvement.

## V2 Predictions

| model_name | batch_size | input_tokens | optimizer_name | experiment_type | actual_peak_allocated_MB | predicted_peak_allocated_MB | allocated_relative_error_percent | actual_peak_reserved_MB | predicted_peak_reserved_MB | reserved_relative_error_percent |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 1 | 32 | adamw | distilgpt2_sequence_length_variation | 1592.43 | 1512.989686 | 4.988622 | 1700.0 | 1634.028860 | 3.880655 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_sequence_length_variation | 1600.69 | 1552.986061 | 2.980211 | 1738.0 | 1677.224945 | 3.496839 |
| distilgpt2 | 1 | 128 | adamw | distilgpt2_sequence_length_variation | 1616.71 | 1632.978811 | 1.006291 | 1726.0 | 1763.617115 | 2.179439 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_batch_size_variation | 1600.19 | 1552.986061 | 2.949896 | 1738.0 | 1677.224945 | 3.496839 |
| distilgpt2 | 2 | 64 | adamw | distilgpt2_batch_size_variation | 1616.21 | 1632.978811 | 1.037539 | 1726.0 | 1763.617115 | 2.179439 |
| distilgpt2 | 4 | 64 | adamw | distilgpt2_batch_size_variation | 1648.43 | 1792.964311 | 8.767998 | 1768.0 | 1936.401455 | 9.524969 |
| distilgpt2 | 1 | 64 | sgd | distilgpt2_optimizer_comparison | 956.49 | 928.042701 | 2.974134 | 1030.0 | 1002.286117 | 2.690668 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_optimizer_comparison | 1600.94 | 1552.986061 | 2.995361 | 1738.0 | 1677.224945 | 3.496839 |

## Row-wise V1 vs V2 Comparison

| model_name | batch_size | input_tokens | optimizer_name | experiment_type | actual_peak_allocated_MB | v1_predicted_allocated_MB | v1_allocated_error_percent | v2_predicted_allocated_MB | v2_allocated_error_percent | allocated_error_improvement_percent_points | actual_peak_reserved_MB | v1_reserved_error_percent | v2_reserved_error_percent | reserved_error_improvement_percent_points |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 1 | 32 | adamw | distilgpt2_sequence_length_variation | 1592.43 | 1309.883094 | 17.743129 | 1512.989686 | 4.988622 | 12.754507 | 1700.0 | 16.783898 | 3.880655 | 12.903242 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_sequence_length_variation | 1600.69 | 1349.879469 | 15.668901 | 1552.986061 | 2.980211 | 12.688690 | 1738.0 | 16.117962 | 3.496839 | 12.621123 |
| distilgpt2 | 1 | 128 | adamw | distilgpt2_sequence_length_variation | 1616.71 | 1429.872219 | 11.556666 | 1632.978811 | 1.006291 | 10.550375 | 1726.0 | 10.529432 | 2.179439 | 8.349994 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_batch_size_variation | 1600.19 | 1349.879469 | 15.642551 | 1552.986061 | 2.949896 | 12.692655 | 1738.0 | 16.117962 | 3.496839 | 12.621123 |
| distilgpt2 | 2 | 64 | adamw | distilgpt2_batch_size_variation | 1616.21 | 1429.872219 | 11.529305 | 1632.978811 | 1.037539 | 10.491766 | 1726.0 | 10.529432 | 2.179439 | 8.349994 |
| distilgpt2 | 4 | 64 | adamw | distilgpt2_batch_size_variation | 1648.43 | 1589.857719 | 3.553216 | 1792.964311 | 8.767998 | -5.214782 | 1768.0 | 2.881995 | 9.524969 | -6.642975 |
| distilgpt2 | 1 | 64 | sgd | distilgpt2_optimizer_comparison | 956.49 | 724.936109 | 24.208710 | 928.042701 | 2.974134 | 21.234576 | 1030.0 | 23.987282 | 2.690668 | 21.296614 |
| distilgpt2 | 1 | 64 | adamw | distilgpt2_optimizer_comparison | 1600.94 | 1349.879469 | 15.682070 | 1552.986061 | 2.995361 | 12.686709 | 1738.0 | 16.117962 | 3.496839 | 12.621123 |

## Row-wise Interpretation

V2 improved almost every row.

The largest improvement was the SGD row:

- V1 allocated error = 24.21%
- V2 allocated error = 2.97%

This proves that backward temporary memory was the missing component.

The only row that got worse was:

- batch_size = 4
- input_tokens = 64
- optimizer = AdamW

For this row:

- V1 allocated error = 3.55%
- V2 allocated error = 8.77%

This happened because V1 was already close on this row, and V2 added a fixed backward temporary correction, causing overprediction.

This is acceptable because overall error improved strongly.

## V2 Error by Optimizer

| optimizer_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 7 | 3.532274 | 2.980211 | 8.767998 | 4.036431 | 3.496839 | 9.524969 | 1610.80 | 1604.409971 | 1733.428571 | 1732.762769 |
| sgd | 1 | 2.974134 | 2.974134 | 2.974134 | 2.690668 | 2.690668 | 2.690668 | 956.49 | 928.042701 | 1030.000000 | 1002.286117 |

## Optimizer-wise Interpretation

V2 handles both optimizers well.

AdamW:

- allocated mean error = 3.53%
- reserved mean error = 4.04%

SGD:

- allocated error = 2.97%
- reserved error = 2.69%

This is a strong improvement over V1.

## V2 Error by Experiment

| experiment_type | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2_batch_size_variation | 3 | 4.251811 | 2.949896 | 8.767998 | 5.067082 | 3.496839 | 9.524969 | 1621.610000 | 1659.643061 | 1744.000000 | 1792.414505 |
| distilgpt2_optimizer_comparison | 2 | 2.984748 | 2.984748 | 2.995361 | 3.093753 | 3.093753 | 3.496839 | 1278.715000 | 1240.514381 | 1384.000000 | 1339.755531 |
| distilgpt2_sequence_length_variation | 3 | 2.991708 | 2.980211 | 4.988622 | 3.185644 | 3.496839 | 3.880655 | 1603.276667 | 1566.318186 | 1721.333333 | 1691.623640 |

## Experiment-wise Interpretation

V2 performs well across all experiment groups.

Best group:

- optimizer comparison
- allocated mean error = 2.98%

Sequence-length variation:

- allocated mean error = 2.99%

Batch-size variation:

- allocated mean error = 4.25%

Batch-size variation is slightly weaker because batch size 4 was overpredicted.

Still, all mean errors are below 6%, which is strong for this stage.

## V2 Component Summary

| optimizer_name | parameter_memory_MB | gradient_memory_MB | optimizer_state_memory_MB | activation_memory_MB | backward_temp_memory_MB | framework_overhead_MB |
|---|---:|---:|---:|---:|---:|---:|
| adamw | 312.471680 | 312.471680 | 624.943359 | 131.416661 | 203.106592 | 20.0 |
| sgd | 312.471680 | 312.471680 | 0.000000 | 79.992750 | 203.106592 | 20.0 |

## Component Interpretation

V2 includes the following training memory components:

- parameter memory
- gradient memory
- optimizer-state memory
- activation memory
- backward temporary memory
- framework overhead
- allocator padding

For AdamW:

- optimizer-state memory = 624.94 MB
- backward temporary memory = 203.11 MB

For SGD:

- optimizer-state memory = 0 MB
- backward temporary memory = 203.11 MB

This explains why AdamW prediction is higher than SGD while both still receive backward temporary correction.

## Main Interpretation

Day 28 significantly improved the TrainingMemoryEstimator.

V1 underpredicted because it missed backward temporary memory.

V2 adds a simple, explainable backward temporary memory term based on parameter memory.

This reduced allocated MRE from 15.66% to 2.98%.

It reduced reserved MRE from 16.12% to 3.50%.

The estimator now predicts distilgpt2 training memory well across sequence-length, batch-size, and optimizer-comparison experiments.

## Important Tradeoff

V2 is not perfect.

It improves overall accuracy, but it overpredicts the batch-size-4 AdamW row.

This is because the backward temporary correction is currently fixed by model size and does not adapt to batch-specific behavior.

This is acceptable for now because the overall error improved strongly.

Future improvement could make backward_temp_memory depend on batch size or observed stage behavior.

## Why This Matters for the Project

Day 28 turns the training estimator from a rough baseline into a strong estimator.

The project now has:

- inference estimator
- optimizer-state estimator
- training memory estimator
- allocator-aware reserved-memory prediction
- measured error metrics

This makes the project much more complete.

## Questions Answered

### Q1. Why did V1 underpredict training memory?

V1 underpredicted because it missed backward temporary memory.

Training creates temporary tensors during backward execution.

These tensors are not fully captured by parameters, gradients, optimizer states, activations, and simple framework overhead.

### Q2. What new component did V2 add?

V2 added:

backward_temp_memory_MB = parameter_memory_MB x backward_temp_factor

with:

backward_temp_factor = 0.65

For distilgpt2, this added 203.11 MB.

### Q3. Did V2 improve allocated MRE?

Yes.

Allocated MRE improved from:

15.66% to 2.98%

### Q4. Did V2 improve reserved MRE?

Yes.

Reserved MRE improved from:

16.12% to 3.50%

### Q5. Did V2 overpredict any rows after adding backward temp memory?

Yes.

The main overprediction happened for:

- batch_size = 4
- input_tokens = 64
- optimizer = AdamW

Actual peak allocated memory:

1648.43 MB

V2 predicted peak allocated memory:

1792.96 MB

Error:

8.77%

This happened because the fixed backward temporary correction overcorrected a row that was already close in V1.

## Day 28 Conclusion

Day 28 improved the TrainingMemoryEstimator by adding a backward temporary memory correction term.

V2 significantly outperformed V1.

Allocated MRE decreased from 15.66% to 2.98%.

Reserved MRE decreased from 16.12% to 3.50%.

Mean allocated error decreased from 14.45% to 3.46%.

Mean reserved error decreased from 14.13% to 3.87%.

The SGD row improved from 24.21% allocated error to 2.97% allocated error, showing that backward temporary memory was the missing component.

AdamW rows also improved, with mean allocated error of 3.53%.

The only notable tradeoff was the batch-size-4 AdamW row, where V2 overpredicted memory and error increased from 3.55% to 8.77%.

Overall, V2 is much stronger and more reliable than V1.

The next step is Day 29: training PEF-style fit/fail simulation.

---


# Day 29 - Training PEF-style Fit/Fail Simulation

## Goal

Evaluate whether TrainingMemoryEstimator V2 can predict whether a training workload fits inside a given GPU memory limit.

Day 28 improved the training estimator and reduced prediction error.

Day 29 tested practical fit/fail behavior using a PEF-style simulation.

This moves beyond numeric prediction error and checks whether the estimator can make scheduling-style decisions.

## Files Created

- results/day29_training_pef_cases.csv
- results/day29_training_pef_summary.csv
- results/day29_training_pef_by_limit.csv
- results/day29_training_pef_by_optimizer.csv
- results/day29_training_pef_by_experiment.csv
- results/day29_training_pef_dangerous_failures.csv
- results/day29_training_pef_conservative_failures.csv
- results/day29_training_realistic_gpu_limit_summary.csv
- report/day29_training_pef_simulation.md
- plots/day29_training_pef_accuracy_by_limit.png
- plots/day29_training_pef_dangerous_failure_by_limit.png
- plots/day29_training_pef_conservative_failure_by_limit.png
- plots/day29_training_pef_accuracy_by_optimizer.png

## Simulation Method

For each workload and memory limit, the simulation compared:

actual_peak_reserved_MB

and

predicted_peak_reserved_MB

The fit/fail decision was made as:

actual_fits = actual_peak_reserved_MB <= memory_limit_MB

predicted_fits = predicted_peak_reserved_MB <= memory_limit_MB

Reserved memory was used because it better reflects PyTorch CUDA allocator behavior and real GPU memory pressure.

## Memory Limits Tested

The following memory limits were tested:

- 512 MB
- 768 MB
- 1024 MB
- 1200 MB
- 1400 MB
- 1500 MB
- 1600 MB
- 1700 MB
- 1725 MB
- 1750 MB
- 1800 MB
- 2048 MB
- 4096 MB
- 8192 MB

These limits include both small artificial limits and realistic GPU memory limits.

## Failure Definitions

### Correct Prediction

A prediction is correct when actual_fits and predicted_fits match.

### Dangerous Failure

A dangerous failure happens when:

predicted_fits = True

but

actual_fits = False

This means the estimator predicts that the workload will fit, but the actual memory exceeds the limit.

This can cause OOM.

### Conservative Failure

A conservative failure happens when:

predicted_fits = False

but

actual_fits = True

This means the estimator rejects a workload that would actually fit.

This wastes GPU capacity, but it is safer than dangerous underprediction.

## Overall Summary

| total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|
| 112 | 102 | 7 | 3 | 91.071429 | 6.25 | 2.678571 |

## Overall Interpretation

The TrainingMemoryEstimator V2 achieved:

- 91.07% fit/fail accuracy
- 6.25% dangerous failure rate
- 2.68% conservative failure rate

This is useful, but not perfect.

The dangerous failure rate is the main weakness because dangerous failures can lead to OOM.

## Summary by Memory Limit

| memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 768 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 1024 | 8 | 7 | 1 | 0 | 87.5 | 12.5 | 0.0 |
| 1200 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 1400 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 1500 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 1600 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 1700 | 8 | 5 | 3 | 0 | 62.5 | 37.5 | 0.0 |
| 1725 | 8 | 5 | 3 | 0 | 62.5 | 37.5 | 0.0 |
| 1750 | 8 | 6 | 0 | 2 | 75.0 | 0.0 | 25.0 |
| 1800 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 2048 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 4096 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 8192 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |

## Memory-limit Interpretation

The estimator worked perfectly at many limits:

- 512 MB
- 768 MB
- 1200 MB
- 1400 MB
- 1500 MB
- 1600 MB
- 2048 MB
- 4096 MB
- 8192 MB

The weak region was around:

1700 MB to 1800 MB

This is the decision-boundary region for the current training workloads.

At 1700 MB and 1725 MB, the estimator had dangerous underpredictions.

At 1750 MB and 1800 MB, the estimator had conservative overpredictions.

This shows that fit/fail prediction is hardest near the actual memory boundary.

## Summary by Optimizer

| optimizer_name | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| adamw | 98 | 89 | 6 | 3 | 90.816327 | 6.122449 | 3.061224 |
| sgd | 14 | 13 | 1 | 0 | 92.857143 | 7.142857 | 0.000000 |

## Optimizer-wise Interpretation

The estimator behaved similarly for AdamW and SGD.

AdamW:

- accuracy = 90.82%
- dangerous failure rate = 6.12%
- conservative failure rate = 3.06%

SGD:

- accuracy = 92.86%
- dangerous failure rate = 7.14%
- conservative failure rate = 0%

AdamW has more cases, so its result is more meaningful than the SGD result.

## Summary by Experiment Type

| experiment_type | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2_batch_size_variation | 42 | 38 | 2 | 2 | 90.476190 | 4.761905 | 4.761905 |
| distilgpt2_optimizer_comparison | 28 | 25 | 3 | 0 | 89.285714 | 10.714286 | 0.000000 |
| distilgpt2_sequence_length_variation | 42 | 39 | 2 | 1 | 92.857143 | 4.761905 | 2.380952 |

## Experiment-wise Interpretation

The best group was sequence-length variation:

- accuracy = 92.86%

The weakest group was optimizer comparison:

- accuracy = 89.29%
- dangerous failure rate = 10.71%

This happened because the optimizer-comparison group included boundary cases.

## Dangerous Failures

| model_name | batch_size | input_tokens | optimizer_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB | actual_fits | predicted_fits | outcome |
|---|---:|---:|---|---:|---:|---:|---|---|---|
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | sgd | 1024 | 1030.0 | 1002.286117 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | False | True | dangerous_underprediction |

## Dangerous Failure Interpretation

There were 7 dangerous failures.

Most dangerous failures came from the repeated AdamW case:

- actual reserved memory = 1738 MB
- predicted reserved memory = 1677.22 MB

At memory limits of 1700 MB and 1725 MB, the estimator predicted that the workload would fit, but actual memory exceeded the limit.

There was also one SGD dangerous failure:

- actual reserved memory = 1030 MB
- predicted reserved memory = 1002.29 MB
- memory limit = 1024 MB

These are boundary cases.

The estimator is close, but when the limit is very near the actual memory, small prediction errors can flip the fit/fail decision.

## Conservative Failures

| model_name | batch_size | input_tokens | optimizer_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB | actual_fits | predicted_fits | outcome |
|---|---:|---:|---|---:|---:|---:|---|---|---|
| distilgpt2 | 1 | 128 | adamw | 1750 | 1726.0 | 1763.617115 | True | False | conservative_overprediction |
| distilgpt2 | 2 | 64 | adamw | 1750 | 1726.0 | 1763.617115 | True | False | conservative_overprediction |
| distilgpt2 | 4 | 64 | adamw | 1800 | 1768.0 | 1936.401455 | True | False | conservative_overprediction |

## Conservative Failure Interpretation

There were 3 conservative failures.

These happened when the estimator predicted slightly higher reserved memory than actual.

This is safer than dangerous underprediction because it avoids OOM, but it can waste GPU capacity by rejecting workloads that would have fit.

## Realistic GPU Limit Summary

| memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent |
|---:|---:|---:|---:|---:|---:|
| 2048 | 8 | 8 | 0 | 0 | 100.0 |
| 4096 | 8 | 8 | 0 | 0 | 100.0 |
| 8192 | 8 | 8 | 0 | 0 | 100.0 |

## Realistic Limit Interpretation

At realistic GPU limits of 2048 MB and above, the estimator achieved:

- 100% accuracy
- 0 dangerous failures
- 0 conservative failures

This means the estimator is reliable for the current workloads at realistic GPU memory limits.

The errors mainly appear near artificial tight limits around 1700–1800 MB.

## Main Interpretation

Day 29 showed that TrainingMemoryEstimator V2 is useful for scheduling-style fit/fail decisions.

The overall accuracy was 91.07%.

The estimator worked perfectly at realistic limits of 2048 MB, 4096 MB, and 8192 MB.

The main weakness is boundary-region behavior.

When memory limits are very close to actual reserved memory, small prediction errors can cause dangerous or conservative failures.

The dangerous failures suggest that a safety margin should be added to predicted reserved memory for scheduling decisions.

## Why This Matters for the Project

Numeric MRE alone is not enough for memory prediction.

A memory estimator should also be judged by whether it can correctly predict fit/fail under memory limits.

Day 29 adds this scheduling-style evaluation to the training estimator.

This makes the project closer to the xMem-style goal of predicting memory requirements for workload placement.

## Questions Answered

### Q1. What is a dangerous failure?

A dangerous failure happens when:

predicted_fits = True

but

actual_fits = False

This means the estimator predicts that the workload will fit, but it actually exceeds the memory limit.

This can cause OOM.

### Q2. What is a conservative failure?

A conservative failure happens when:

predicted_fits = False

but

actual_fits = True

This means the estimator rejects a workload that would actually fit.

This wastes GPU capacity, but it is safer than dangerous underprediction.

### Q3. What was the overall PEF accuracy?

Overall PEF accuracy was:

91.07%

There were 102 correct predictions out of 112 total cases.

### Q4. Were there any dangerous failures?

Yes.

There were 7 dangerous failures.

The dangerous failure rate was:

6.25%

Most dangerous failures happened around 1700 MB and 1725 MB memory limits.

### Q5. How did the estimator behave at realistic limits like 2048 MB, 4096 MB, and 8192 MB?

The estimator performed perfectly at these realistic limits.

At 2048 MB, 4096 MB, and 8192 MB:

- accuracy = 100%
- dangerous failures = 0
- conservative failures = 0

## Day 29 Conclusion

Day 29 completed training PEF-style simulation.

TrainingMemoryEstimator V2 achieved 91.07% fit/fail accuracy across 112 simulated cases.

The dangerous failure rate was 6.25%.

The conservative failure rate was 2.68%.

The estimator performed perfectly at realistic larger GPU limits of 2048 MB, 4096 MB, and 8192 MB.

Most errors occurred near the memory boundary region around 1700–1800 MB.

At 1700 MB and 1725 MB, the estimator produced dangerous underpredictions for workloads whose actual reserved memory was 1738 MB but predicted reserved memory was 1677.22 MB.

At 1750 MB and 1800 MB, the estimator produced conservative overpredictions for workloads where predicted reserved memory was slightly above the limit but actual memory still fit.

This shows that the estimator is useful for fit/fail prediction, especially away from tight boundary regions, but boundary-region safety could be improved by adding a safety margin to predicted reserved memory.

The next step is Day 30: add safety-margin PEF simulation or prepare a training phase report.

---


# Day 30 - Training PEF Safety-margin Simulation

## Goal

Evaluate whether adding a safety margin to predicted reserved memory reduces dangerous fit/fail failures.

Day 29 showed that TrainingMemoryEstimator V2 achieved 91.07% PEF accuracy, but still had dangerous failures near tight memory limits.

Day 30 tested safety margins to reduce dangerous underprediction.

## Files Created

- results/day30_training_pef_safety_margin_cases.csv
- results/day30_training_pef_safety_margin_summary.csv
- results/day30_training_pef_safety_margin_by_limit.csv
- results/day30_training_pef_safety_margin_by_optimizer.csv
- results/day30_training_pef_safety_margin_dangerous_failures.csv
- results/day30_training_pef_safety_margin_conservative_failures.csv
- results/day30_training_pef_safety_margin_realistic_limits.csv
- results/day30_training_pef_best_safety_margin.csv
- report/day30_training_pef_safety_margin.md
- plots/day30_dangerous_failure_vs_safety_margin.png
- plots/day30_conservative_failure_vs_safety_margin.png
- plots/day30_accuracy_vs_safety_margin.png
- plots/day30_pef_tradeoff_vs_safety_margin.png

## Safety-margin Method

The safety-margin method increases predicted reserved memory before making a fit/fail decision.

Formula:

safe_predicted_reserved_MB =
predicted_peak_reserved_MB x (1 + safety_margin)

The tested safety margins were:

- 0%
- 5%
- 10%
- 15%

## Why Safety Margin Was Needed

Day 29 showed that the estimator had dangerous failures near memory-boundary limits.

A dangerous failure happens when:

predicted_fits = True

but

actual_fits = False

This can cause OOM.

A safety margin makes the estimator more cautious by slightly increasing the predicted reserved memory.

## Summary by Safety Margin

| safety_margin | safety_margin_percent | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0 | 112 | 102 | 7 | 3 | 91.071429 | 6.25 | 2.678571 |
| 0.05 | 5.0 | 112 | 103 | 0 | 9 | 91.964286 | 0.00 | 8.035714 |
| 0.10 | 10.0 | 112 | 97 | 0 | 15 | 86.607143 | 0.00 | 13.392857 |
| 0.15 | 15.0 | 112 | 96 | 0 | 16 | 85.714286 | 0.00 | 14.285714 |

## Main Result

The 5% safety margin gave the best tradeoff.

Without safety margin:

- accuracy = 91.07%
- dangerous failures = 7
- conservative failures = 3

With 5% safety margin:

- accuracy = 91.96%
- dangerous failures = 0
- conservative failures = 9

The 5% margin removed all dangerous failures and slightly improved accuracy.

Larger margins of 10% and 15% also removed dangerous failures, but they increased conservative failures too much and reduced accuracy.

## Best Safety Margin Candidate

| safety_margin | safety_margin_percent | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5.0 | 112 | 103 | 0 | 9 | 91.964286 | 0.0 | 8.035714 |

## Why 5% Was Selected

The primary goal is to reduce dangerous failures.

A dangerous failure can cause OOM.

A conservative failure only rejects a workload that would have fit, which wastes capacity but is safer.

The 5% margin removed all dangerous failures while keeping accuracy highest among all tested margins.

So 5% is the best safety-margin choice for the current dataset.

## Summary by Memory Limit

| safety_margin_percent | memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 512 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 768 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 1024 | 8 | 7 | 1 | 0 | 87.5 | 12.5 | 0.0 |
| 0.0 | 1200 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 1400 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 1500 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 1600 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 1700 | 8 | 5 | 3 | 0 | 62.5 | 37.5 | 0.0 |
| 0.0 | 1725 | 8 | 5 | 3 | 0 | 62.5 | 37.5 | 0.0 |
| 0.0 | 1750 | 8 | 6 | 0 | 2 | 75.0 | 0.0 | 25.0 |
| 0.0 | 1800 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 0.0 | 2048 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 0.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 512 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 768 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1024 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1200 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1400 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1500 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1600 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1700 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 5.0 | 1725 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 1750 | 8 | 3 | 0 | 5 | 37.5 | 0.0 | 62.5 |
| 5.0 | 1800 | 8 | 5 | 0 | 3 | 62.5 | 0.0 | 37.5 |
| 5.0 | 2048 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 5.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 512 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 768 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1024 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1200 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1400 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1500 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1600 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 1700 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 10.0 | 1725 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 10.0 | 1750 | 8 | 2 | 0 | 6 | 25.0 | 0.0 | 75.0 |
| 10.0 | 1800 | 8 | 2 | 0 | 6 | 25.0 | 0.0 | 75.0 |
| 10.0 | 2048 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 10.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 10.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 512 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 768 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1024 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1200 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1400 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1500 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1600 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 1700 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 15.0 | 1725 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 15.0 | 1750 | 8 | 2 | 0 | 6 | 25.0 | 0.0 | 75.0 |
| 15.0 | 1800 | 8 | 1 | 0 | 7 | 12.5 | 0.0 | 87.5 |
| 15.0 | 2048 | 8 | 7 | 0 | 1 | 87.5 | 0.0 | 12.5 |
| 15.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| 15.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 | 0.0 | 0.0 |

## Memory-limit Interpretation

The 5% margin removed dangerous failures at boundary limits like 1700 MB and 1725 MB.

However, it increased conservative failures at 1750 MB and 1800 MB.

This shows the expected tradeoff:

- safety margin reduces OOM risk
- safety margin increases cautious rejection

## Summary by Optimizer

| safety_margin_percent | optimizer_name | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | adamw | 98 | 89 | 6 | 3 | 90.816327 | 6.122449 | 3.061224 |
| 0.0 | sgd | 14 | 13 | 1 | 0 | 92.857143 | 7.142857 | 0.000000 |
| 5.0 | adamw | 98 | 89 | 0 | 9 | 90.816327 | 0.000000 | 9.183673 |
| 5.0 | sgd | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 10.0 | adamw | 98 | 83 | 0 | 15 | 84.693878 | 0.000000 | 15.306122 |
| 10.0 | sgd | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 15.0 | adamw | 98 | 82 | 0 | 16 | 83.673469 | 0.000000 | 16.326531 |
| 15.0 | sgd | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |

## Optimizer-wise Interpretation

For AdamW, 5% safety margin removed dangerous failures without reducing accuracy compared to no margin.

For SGD, 5% safety margin improved fit/fail accuracy to 100% for the tested cases.

Larger margins became overly conservative for AdamW.

## Remaining Dangerous Failures

| safety_margin_percent | model_name | batch_size | input_tokens | optimizer_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB | safe_predicted_reserved_MB | actual_fits | predicted_fits | outcome |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | sgd | 1024 | 1030.0 | 1002.286117 | 1002.286117 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |
| 0.0 | distilgpt2 | 1 | 64 | adamw | 1725 | 1738.0 | 1677.224945 | 1677.224945 | False | True | dangerous_underprediction |

## Dangerous Failure Interpretation

All dangerous failures occurred only at 0% margin.

After adding 5% margin, dangerous failures became zero.

This means the safety margin successfully removed dangerous underprediction.

## Realistic GPU Limit Summary

| safety_margin_percent | memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 2048 | 8 | 8 | 0 | 0 | 100.0 |
| 0.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 |
| 0.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 |
| 5.0 | 2048 | 8 | 8 | 0 | 0 | 100.0 |
| 5.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 |
| 5.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 |
| 10.0 | 2048 | 8 | 7 | 0 | 1 | 87.5 |
| 10.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 |
| 10.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 |
| 15.0 | 2048 | 8 | 7 | 0 | 1 | 87.5 |
| 15.0 | 4096 | 8 | 8 | 0 | 0 | 100.0 |
| 15.0 | 8192 | 8 | 8 | 0 | 0 | 100.0 |

## Realistic Limit Interpretation

At realistic GPU limits of 2048 MB, 4096 MB, and 8192 MB, the 5% safety margin achieved:

- 100% accuracy
- 0 dangerous failures
- 0 conservative failures

This means 5% margin is safe for current realistic workload placement.

10% and 15% margins became unnecessarily conservative at 2048 MB.

## Main Interpretation

Day 30 showed that a small safety margin can make the training estimator safer for scheduling decisions.

The best result came from the 5% margin.

The 5% margin removed all dangerous failures and slightly improved overall accuracy.

Larger margins also removed dangerous failures, but they became too conservative and reduced accuracy.

## Why This Matters for the Project

MRE alone is not enough for a memory estimator.

A scheduler cares about whether a workload will fit or OOM.

Day 30 shows that TrainingMemoryEstimator V2 can be made safer by adding a 5% reserved-memory safety margin.

This is directly useful for fit/fail prediction.

## Questions Answered

### Q1. Did safety margin reduce dangerous failures?

Yes.

Dangerous failures changed as follows:

- 0% margin: 7 dangerous failures
- 5% margin: 0 dangerous failures
- 10% margin: 0 dangerous failures
- 15% margin: 0 dangerous failures

### Q2. Which safety margin gave the best tradeoff?

The 5% safety margin gave the best tradeoff.

It removed all dangerous failures and gave the highest overall accuracy:

91.96%

### Q3. Did conservative failures increase?

Yes.

Conservative failures increased as safety margin increased:

- 0% margin: 3 conservative failures
- 5% margin: 9 conservative failures
- 10% margin: 15 conservative failures
- 15% margin: 16 conservative failures

This is expected because the estimator becomes more cautious.

### Q4. What happened to overall accuracy?

Overall accuracy changed as follows:

- 0% margin: 91.07%
- 5% margin: 91.96%
- 10% margin: 86.61%
- 15% margin: 85.71%

Accuracy improved slightly at 5%, then dropped at larger margins because conservative failures increased.

### Q5. Why is a safety margin useful for scheduling?

A safety margin is useful because prediction errors near memory limits can cause OOM.

Adding a safety margin makes the estimator more cautious.

This reduces dangerous underprediction.

For scheduling, this is important because dangerous failure is worse than conservative rejection.

## Day 30 Conclusion

Day 30 completed safety-margin PEF simulation.

The 5% safety margin was the best choice.

Without safety margin, TrainingMemoryEstimator V2 had:

- 91.07% accuracy
- 7 dangerous failures
- 3 conservative failures

With 5% safety margin, it had:

- 91.96% accuracy
- 0 dangerous failures
- 9 conservative failures

The 5% margin removed all dangerous failures and slightly improved accuracy.

Larger margins of 10% and 15% also removed dangerous failures, but they increased conservative failures too much and reduced accuracy.

At realistic GPU limits of 2048 MB, 4096 MB, and 8192 MB, the 5% safety margin achieved 100% accuracy with zero dangerous and zero conservative failures.

Therefore, 5% safety margin is the best choice for the current dataset.

The next step is Day 31: training phase report and clean GitHub update.

---

# Day 31 - Training Phase Report and Clean GitHub Preparation

## Goal

Create a clean training phase report and prepare GitHub-facing files without day-numbered filenames.

## Clean Files Created

- report/training_phase_report.md
- report/readme_training_section_draft.md
- report/github_training_phase_checklist.md
- results/training_phase_key_metrics.csv
- results/training_phase_key_findings.csv

## Clean GitHub-facing Copies Created

Reports:

- report/optimizer_state_estimator_report.md
- report/training_memory_estimator_report.md
- report/training_estimator_improvement_report.md
- report/training_pef_simulation_report.md
- report/training_pef_safety_margin_report.md

Results:

- results/training_estimator_v2_predictions.csv
- results/training_estimator_v2_metrics.csv
- results/training_estimator_v1_vs_v2_metrics.csv
- results/training_pef_summary.csv
- results/training_pef_safety_margin_summary.csv
- results/training_pef_best_safety_margin.csv

Plots:

- plots/training_estimator_v1_vs_v2_allocated_mre.png
- plots/training_estimator_v1_vs_v2_reserved_mre.png
- plots/training_actual_vs_predicted_allocated.png
- plots/training_pef_dangerous_failure_vs_safety_margin.png
- plots/training_pef_accuracy_vs_safety_margin.png
- plots/training_pef_tradeoff_vs_safety_margin.png

## Main Interpretation

The training phase is now organized into clean GitHub-facing files.

Day-wise files remain private and should not be pushed.

The GitHub repo should show project-level reports and results, not daily progress logs.

## Day 31 Conclusion

Day 31 created clean training phase documentation and GitHub-ready file names.

The next step is to push this update to GitHub and then start gpt2 validation.

---

# Day 32 - gpt2 Inference Validation

## Goal

Validate whether the inference memory estimator generalizes from distilgpt2 to gpt2.

Most previous meaningful inference validation was on distilgpt2. Day 32 tested the estimator on a larger GPT-style model.

## Experiment Setup

- model = gpt2
- task = inference
- GPU = Tesla T4
- batch sizes = 1, 2
- input tokens = 64, 128, 256
- max_new_tokens = 32, 128
- dtype = fp32
- one fp16 sanity run
- use_cache = True

## Files Created

- results/gpt2_inference_validation.csv
- results/gpt2_inference_estimator_predictions.csv
- results/gpt2_inference_estimator_metrics.csv
- results/gpt2_inference_error_by_dtype.csv
- results/gpt2_inference_error_by_batch_size.csv
- results/model_generalization_inference_metrics.csv
- plots/gpt2_inference_actual_vs_predicted_allocated.png
- plots/gpt2_inference_actual_vs_predicted_reserved.png
- report/gpt2_inference_validation_report.md

## Profiling Results

| model_name | batch_size | input_tokens | max_new_tokens | dtype | use_cache | oom | peak_allocated_MB | peak_reserved_MB | runtime_sec |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| gpt2 | 1 | 64 | 32 | fp32 | True | False | 493.84 | 540.0 | 1.7537 |
| gpt2 | 1 | 64 | 128 | fp32 | True | False | 499.42 | 550.0 | 3.2135 |
| gpt2 | 1 | 128 | 32 | fp32 | True | False | 503.97 | 546.0 | 0.5285 |
| gpt2 | 1 | 128 | 128 | fp32 | True | False | 504.11 | 558.0 | 2.0182 |
| gpt2 | 1 | 256 | 32 | fp32 | True | False | 519.72 | 582.0 | 0.3335 |
| gpt2 | 1 | 256 | 128 | fp32 | True | False | 532.72 | 582.0 | 2.8287 |
| gpt2 | 2 | 64 | 32 | fp32 | True | False | 503.97 | 550.0 | 0.7964 |
| gpt2 | 2 | 64 | 128 | fp32 | True | False | 532.94 | 582.0 | 1.8660 |
| gpt2 | 2 | 128 | 32 | fp32 | True | False | 519.72 | 582.0 | 1.3682 |
| gpt2 | 2 | 128 | 128 | fp32 | True | False | 532.95 | 582.0 | 1.3261 |
| gpt2 | 2 | 256 | 32 | fp32 | True | False | 562.73 | 592.0 | 0.3489 |
| gpt2 | 2 | 256 | 128 | fp32 | True | False | 562.73 | 592.0 | 1.2612 |
| gpt2 | 1 | 128 | 128 | fp16 | True | False | 264.78 | 268.0 | 1.4381 |

## Main Memory Observations

All 13 gpt2 inference runs completed successfully with no OOM.

For fp32 gpt2 inference, peak allocated memory ranged from 493.84 MB to 562.73 MB.

This is higher than earlier distilgpt2 inference memory, which was around 331 MB to 384 MB peak allocated memory.

This is expected because gpt2 has more parameters than distilgpt2.

## fp16 Result

For the same setting:

- batch_size = 1
- input_tokens = 128
- max_new_tokens = 128

fp32 peak allocated memory was 504.11 MB.

fp16 peak allocated memory was 264.78 MB.

Memory reduction:

504.11 MB - 264.78 MB = 239.33 MB

Percentage reduction:

around 47.48%

This is consistent with the earlier distilgpt2 fp16 reduction.

## Batch-size Scaling

Batch-size scaling remained sublinear.

Example:

- batch=1, input=256, max_new=128: 532.72 MB peak allocated
- batch=2, input=256, max_new=128: 562.73 MB peak allocated

Doubling batch size increased memory by only 30.01 MB.

This happens because model weights are shared across batch elements.

## Estimator Prediction Results

| batch_size | input_tokens | max_new_tokens | dtype | actual_peak_allocated_MB | predicted_peak_allocated_MB | allocated_error_percent | actual_peak_reserved_MB | predicted_peak_reserved_MB | reserved_error_percent |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 32 | fp32 | 493.84 | 515.116084 | 4.308295 | 540.0 | 560.0 | 3.703704 |
| 1 | 64 | 128 | fp32 | 499.42 | 518.497969 | 3.820025 | 550.0 | 564.0 | 2.545455 |
| 1 | 128 | 32 | fp32 | 503.97 | 517.370674 | 2.659022 | 546.0 | 562.0 | 2.930403 |
| 1 | 128 | 128 | fp32 | 504.11 | 520.752559 | 3.301375 | 558.0 | 566.0 | 1.433692 |
| 1 | 256 | 32 | fp32 | 519.72 | 521.879854 | 0.415580 | 582.0 | 568.0 | 2.405498 |
| 1 | 256 | 128 | fp32 | 532.72 | 525.261739 | 1.400034 | 582.0 | 572.0 | 1.718213 |
| 2 | 64 | 32 | fp32 | 503.97 | 520.537969 | 3.287491 | 550.0 | 566.0 | 2.909091 |
| 2 | 64 | 128 | fp32 | 532.94 | 527.301739 | 1.057954 | 582.0 | 574.0 | 1.374570 |
| 2 | 128 | 32 | fp32 | 519.72 | 525.047149 | 1.025004 | 582.0 | 570.0 | 2.061856 |
| 2 | 128 | 128 | fp32 | 532.95 | 531.810919 | 0.213731 | 582.0 | 578.0 | 0.687285 |
| 2 | 256 | 32 | fp32 | 562.73 | 534.065509 | 5.093827 | 592.0 | 580.0 | 2.027027 |
| 2 | 256 | 128 | fp32 | 562.73 | 540.829279 | 3.891870 | 592.0 | 588.0 | 0.675676 |
| 1 | 128 | 128 | fp16 | 264.78 | 259.438925 | 2.017175 | 268.0 | 282.0 | 5.223881 |

## gpt2 Global Estimator Metrics

| model_name | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 13 | 2.659022 | 2.499337 | 5.093827 | 0.213731 | 2.061856 | 2.284335 | 5.223881 | 0.675676 | 502.584615 | 504.454644 | 546.615385 | 548.461538 |

## Main Estimator Result

The inference estimator generalized well to gpt2.

Across 13 rows:

- allocated MRE = 2.66%
- allocated mean error = 2.50%
- allocated max error = 5.09%
- reserved MRE = 2.06%
- reserved mean error = 2.28%
- reserved max error = 5.22%

Average actual vs predicted allocated memory was very close:

- average actual allocated = 502.58 MB
- average predicted allocated = 504.45 MB

Average actual vs predicted reserved memory was also very close:

- average actual reserved = 546.62 MB
- average predicted reserved = 548.46 MB

## Error by dtype

| dtype | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 2.017175 | 2.017175 | 2.017175 | 5.223881 | 5.223881 | 5.223881 | 264.780000 | 259.438925 | 268.000000 | 282.000000 |
| fp32 | 12 | 2.539517 | 2.973257 | 5.093827 | 2.039372 | 2.044441 | 3.703704 | 522.401667 | 524.872620 | 569.833333 | 570.666667 |

## Error by Batch Size

| batch_size | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 2.560215 | 2.659022 | 4.308295 | 2.851549 | 2.545455 | 5.223881 | 474.08 | 482.616829 | 518.0 | 524.857143 |
| 2 | 6 | 2.428313 | 2.172723 | 5.093827 | 1.622584 | 1.700799 | 2.909091 | 535.84 | 529.932094 | 580.0 | 576.000000 |

## Model Generalization Comparison

| model_name | phase | allocated_mean_error | reserved_mean_error | note |
|---|---|---:|---:|---|
| distilgpt2 | inference | 2.360000 | 2.360000 | from previous distilgpt2 inference evaluation |
| gpt2 | inference | 2.499337 | 2.284335 | from gpt2 validation |

## Generalization Interpretation

The gpt2 error is very close to the previous distilgpt2 error.

distilgpt2 inference:

- allocated mean error = 2.36%
- reserved mean error = 2.36%

gpt2 inference:

- allocated mean error = 2.50%
- reserved mean error = 2.28%

This supports that the estimator is not overfitted only to distilgpt2.

It generalizes well to a larger GPT-style model.

## Questions Answered

### Q1. Did gpt2 use more memory than distilgpt2?

Yes. gpt2 fp32 inference used around 493 MB to 563 MB peak allocated memory, while distilgpt2 earlier used around 331 MB to 384 MB.

### Q2. Was estimator error close to distilgpt2, or much worse?

The error was close to distilgpt2. gpt2 allocated mean error was 2.50%, compared to 2.36% for distilgpt2.

### Q3. Did fp16 reduce memory for gpt2?

Yes. For batch=1, input=128, max_new=128, fp32 used 504.11 MB peak allocated memory while fp16 used 264.78 MB. This is around 47.48% reduction.

### Q4. Did batch size increase memory sublinearly?

Yes. Doubling batch size from 1 to 2 increased memory much less than 2x because model weights are shared across batch elements.

### Q5. Does this support generalization or show estimator weakness?

It supports generalization for GPT-style inference. The estimator maintained around 2 to 3% error on gpt2, similar to distilgpt2.

## Day 32 Conclusion

Day 32 validated the inference estimator on gpt2.

All 13 runs completed successfully with no OOM.

gpt2 used more memory than distilgpt2, as expected.

The fp16 sanity run reduced memory by around 47.48%.

The inference estimator generalized well to gpt2, achieving 2.66% allocated MRE and 2.06% reserved MRE.

The gpt2 mean errors were very close to the earlier distilgpt2 inference errors, supporting model generalization.

The next step is Day 33: gpt2 training validation.

---

# Day 33 - gpt2 Training Validation

## Goal

Validate whether the training memory estimator generalizes from distilgpt2 to gpt2.

Day 32 showed strong inference generalization to gpt2. Day 33 tests the harder case: training memory prediction.

## Experiment Setup

- model = gpt2
- task = training
- GPU = Tesla T4
- batch_size = 1
- input_tokens = 32, 64, 128
- optimizers = SGD, AdamW
- dtype = fp32

## Files Created

- results/gpt2_training_validation.csv
- results/gpt2_training_stage_memory.csv
- results/gpt2_training_estimator_predictions.csv
- results/gpt2_training_estimator_metrics.csv
- results/gpt2_training_error_by_optimizer.csv
- results/gpt2_training_error_by_sequence_length.csv
- results/model_generalization_training_metrics.csv
- plots/gpt2_training_actual_vs_predicted_allocated.png
- plots/gpt2_training_actual_vs_predicted_reserved.png
- report/gpt2_training_validation_report.md

## Stage-wise Peak Allocated Memory

| optimizer_name | stage | 32 tokens | 64 tokens | 128 tokens |
|---|---|---:|---:|---:|
| adamw | before_model_load | 17.88 | 17.88 | 17.88 |
| adamw | after_model_load | 493.96 | 493.96 | 493.96 |
| adamw | after_batch_creation | 493.96 | 493.96 | 493.97 |
| adamw | after_optimizer_creation | 493.96 | 493.96 | 493.97 |
| adamw | after_forward | 538.55 | 583.11 | 682.99 |
| adamw | after_loss | 538.55 | 583.11 | 682.99 |
| adamw | after_backward | 1271.93 | 1279.20 | 1292.34 |
| adamw | after_optimizer_step | 2402.82 | 2409.59 | 2422.73 |
| adamw | after_zero_grad | 2402.82 | 2409.59 | 2422.73 |
| sgd | before_model_load | 0.00 | 17.88 | 17.88 |
| sgd | after_model_load | 475.46 | 493.96 | 493.96 |
| sgd | after_batch_creation | 475.46 | 493.96 | 493.97 |
| sgd | after_optimizer_creation | 475.46 | 493.96 | 493.97 |
| sgd | after_forward | 529.80 | 583.11 | 682.99 |
| sgd | after_loss | 529.80 | 583.11 | 682.99 |
| sgd | after_backward | 1271.56 | 1279.20 | 1292.34 |
| sgd | after_optimizer_step | 1271.56 | 1279.20 | 1292.34 |
| sgd | after_zero_grad | 1271.56 | 1279.20 | 1292.34 |

## Stage-wise Current Allocated Memory

| optimizer_name | stage | 32 tokens | 64 tokens | 128 tokens |
|---|---|---:|---:|---:|
| adamw | before_model_load | 17.88 | 17.88 | 17.88 |
| adamw | after_model_load | 493.96 | 493.96 | 493.96 |
| adamw | after_batch_creation | 493.96 | 493.96 | 493.97 |
| adamw | after_optimizer_creation | 493.96 | 493.96 | 493.97 |
| adamw | after_forward | 538.55 | 583.11 | 682.99 |
| adamw | after_loss | 538.55 | 583.11 | 682.99 |
| adamw | after_backward | 975.93 | 983.20 | 996.34 |
| adamw | after_optimizer_step | 1927.35 | 1934.12 | 1947.27 |
| adamw | after_zero_grad | 1451.53 | 1457.16 | 1469.43 |
| sgd | before_model_load | 0.00 | 17.88 | 17.88 |
| sgd | after_model_load | 475.46 | 493.96 | 493.96 |
| sgd | after_batch_creation | 475.46 | 493.96 | 493.97 |
| sgd | after_optimizer_creation | 475.46 | 493.96 | 493.97 |
| sgd | after_forward | 529.80 | 583.11 | 682.99 |
| sgd | after_loss | 529.80 | 583.11 | 682.99 |
| sgd | after_backward | 975.56 | 983.20 | 996.34 |
| sgd | after_optimizer_step | 975.56 | 983.20 | 996.34 |
| sgd | after_zero_grad | 499.48 | 506.23 | 518.51 |

## Stage-wise Interpretation

The stage-wise logs show the same optimizer behavior seen earlier with distilgpt2.

For SGD, memory after optimizer_step stays almost the same as after_backward.

For AdamW, memory jumps sharply after optimizer_step.

At 64 tokens:

- AdamW current allocated after_backward = 983.20 MB
- AdamW current allocated after_optimizer_step = 1934.12 MB
- jump = 950.92 MB

This closely matches the estimated AdamW optimizer-state memory of 949.40 MB.

This strongly supports the 2x parameter-memory rule for AdamW optimizer state.

## gpt2 Training Estimator Predictions

| optimizer | input_tokens | actual_peak_allocated_MB | predicted_peak_allocated_MB | allocated_error_percent | actual_peak_reserved_MB | predicted_peak_reserved_MB | reserved_error_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| sgd | 32 | 1271.56 | 1338.717143 | 5.281476 | 1388.0 | 1445.814514 | 4.165311 |
| sgd | 64 | 1279.20 | 1399.478768 | 9.402655 | 1422.0 | 1511.437069 | 6.289527 |
| sgd | 128 | 1292.34 | 1521.002018 | 17.693642 | 1406.0 | 1642.682179 | 16.833725 |
| adamw | 32 | 2402.82 | 2288.117533 | 4.773660 | 2616.0 | 2471.166936 | 5.536432 |
| adamw | 64 | 2409.59 | 2348.879158 | 2.519551 | 2670.0 | 2536.789491 | 4.989158 |
| adamw | 128 | 2422.73 | 2470.402408 | 1.967714 | 2634.0 | 2668.034601 | 1.292126 |

## gpt2 Training Global Metrics

| model_name | phase | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | training | 6 | 5.027568 | 6.939783 | 17.693642 | 1.967714 | 5.262795 | 6.517713 | 16.833725 | 1.292126 | 1846.373333 | 1894.432838 | 2022.666667 | 2045.987465 |

## Main Estimator Result

The training estimator generalized reasonably to gpt2.

Across 6 rows:

- allocated MRE = 5.03%
- allocated mean error = 6.94%
- allocated max error = 17.69%
- reserved MRE = 5.26%
- reserved mean error = 6.52%
- reserved max error = 16.83%

This is worse than distilgpt2, but still acceptable for a larger model.

## Error by Optimizer

| optimizer_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 3 | 3.086975 | 2.519551 | 4.773660 | 3.939239 | 4.989158 | 5.536432 | 2411.713333 | 2369.133033 | 2640.000000 | 2558.663676 |
| sgd | 3 | 10.792591 | 9.402655 | 17.693642 | 9.096188 | 6.289527 | 16.833725 | 1281.033333 | 1419.732643 | 1405.333333 | 1533.311254 |

## Optimizer-wise Interpretation

The estimator works much better for AdamW than SGD.

AdamW:

- allocated mean error = 3.09%
- reserved mean error = 3.94%

SGD:

- allocated mean error = 10.79%
- reserved mean error = 9.10%

The main weakness is SGD overprediction, especially at longer sequence length.

## Error by Sequence Length

| input_tokens | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 5.027568 | 5.027568 | 5.281476 | 4.850871 | 4.850871 | 5.536432 | 1837.190 | 1813.417338 | 2002.0 | 1958.490725 |
| 64 | 2 | 5.961103 | 5.961103 | 9.402655 | 5.639342 | 5.639342 | 6.289527 | 1844.395 | 1874.178963 | 2046.0 | 2024.113280 |
| 128 | 2 | 9.830678 | 9.830678 | 17.693642 | 9.062926 | 9.062926 | 16.833725 | 1857.535 | 1995.702213 | 2020.0 | 2155.358390 |

## Sequence-length Interpretation

Error increases with sequence length.

At 128 tokens, allocated mean error reached 9.83% and reserved mean error reached 9.06%.

This suggests the current activation or backward temporary memory correction may be too aggressive for gpt2 training, especially for SGD.

## Model Generalization Comparison

| model_name | phase | allocated_mean_error | reserved_mean_error | allocated_MRE | reserved_MRE | note |
|---|---|---:|---:|---:|---:|---|
| distilgpt2 | training | 3.462507 | 3.868211 | 2.977173 | 3.496839 | from distilgpt2 TrainingMemoryEstimator V2 |
| gpt2 | training | 6.939783 | 6.517713 | 5.027568 | 5.262795 | from gpt2 training validation |

## Generalization Interpretation

gpt2 training error is higher than distilgpt2 training error.

distilgpt2:

- allocated mean error = 3.46%
- reserved mean error = 3.87%

gpt2:

- allocated mean error = 6.94%
- reserved mean error = 6.52%

This means the estimator generalizes reasonably, but not as cleanly as the inference estimator.

The main weakness is optimizer-specific behavior, especially SGD.

## Questions Answered

### Q1. Did gpt2 training use more memory than distilgpt2 training?

Yes. gpt2 training used more memory.

For gpt2:

- SGD average peak allocated = 1281.03 MB
- AdamW average peak allocated = 2411.71 MB

Earlier distilgpt2 training was around:

- SGD about 956 MB
- AdamW about 1601 MB

### Q2. Did AdamW use much more memory than SGD for gpt2?

Yes. AdamW used much more memory than SGD.

For gpt2:

- AdamW average peak allocated = 2411.71 MB
- SGD average peak allocated = 1281.03 MB
- difference = 1130.68 MB

### Q3. Did the optimizer-state rule still make sense?

Yes. The optimizer-state rule still makes strong sense.

For gpt2:

- parameter memory = 474.70 MB
- AdamW optimizer-state estimate = 949.40 MB

At 64 tokens, AdamW current allocated memory jumped from 983.20 MB after backward to 1934.12 MB after optimizer_step.

The jump was 950.92 MB, which almost exactly matches the predicted optimizer-state memory of 949.40 MB.

### Q4. Was training estimator error close to distilgpt2, or worse?

It was worse than distilgpt2, but still acceptable.

distilgpt2:

- allocated mean error = 3.46%
- reserved mean error = 3.87%

gpt2:

- allocated mean error = 6.94%
- reserved mean error = 6.52%

### Q5. Does this support training generalization or show estimator weakness?

It does both.

It supports generalization because error stayed around 5 to 7%, which is usable.

It also shows weakness because SGD error was much higher than AdamW error, especially at 128 tokens.

## Day 33 Conclusion

Day 33 validated the training memory estimator on gpt2.

All 6 training runs completed successfully with no OOM.

gpt2 training used substantially more memory than distilgpt2 training.

The stage-wise memory logs strongly confirmed AdamW optimizer-state behavior.

At 64 tokens, the AdamW optimizer-step memory jump was 950.92 MB, almost exactly matching the estimated optimizer-state memory of 949.40 MB.

The training estimator generalized reasonably to gpt2, achieving 5.03% allocated MRE and 5.26% reserved MRE.

However, the estimator was weaker on gpt2 than on distilgpt2.

The main weakness was SGD overprediction at longer sequence length.

The next step is Day 34: improve the training estimator using optimizer-specific correction.

---

# Day 34 - TrainingMemoryEstimator V3 Optimizer-specific Correction

## Goal

Improve the training memory estimator by adding optimizer-specific backward temporary memory correction.

Day 33 showed that TrainingMemoryEstimator V2 generalized reasonably to gpt2, but the error was much higher for SGD than AdamW.

V2 used the same backward temporary correction for all optimizers:

backward_temp_factor = 0.65

Day 34 tested whether using a smaller SGD correction improves training-memory prediction.

## V3 Change

TrainingMemoryEstimator V3 uses optimizer-specific backward temporary factors:

- AdamW: 0.65
- Adam: 0.65
- SGD: 0.35
- SGD with momentum: 0.45
- default: 0.50

## Files Created

- src/estimators/training_memory_estimator_v3.py
- results/combined_training_validation_data.csv
- results/training_estimator_v3_predictions.csv
- results/training_estimator_v3_metrics.csv
- results/training_estimator_v3_error_by_model.csv
- results/training_estimator_v3_error_by_optimizer.csv
- results/training_estimator_v3_error_by_sequence_length.csv
- results/training_estimator_v2_vs_v3_metrics.csv
- results/training_estimator_v2_vs_v3_row_comparison.csv
- plots/training_estimator_v2_vs_v3_allocated_mre.png
- plots/training_estimator_v2_vs_v3_reserved_mre.png
- plots/training_estimator_v3_actual_vs_predicted_allocated.png
- plots/training_estimator_v3_actual_vs_predicted_reserved.png
- report/training_estimator_v3_optimizer_correction_report.md

## Combined Training Validation Dataset

The combined validation dataset included both distilgpt2 and gpt2 training rows.

| model_name | batch_size | input_tokens | dtype | optimizer_name | actual_peak_allocated_MB | actual_peak_reserved_MB |
|---|---:|---:|---|---|---:|---:|
| distilgpt2 | 1 | 32 | fp32 | adamw | 1592.43 | 1700.0 |
| distilgpt2 | 1 | 64 | fp32 | adamw | 1600.69 | 1738.0 |
| distilgpt2 | 1 | 128 | fp32 | adamw | 1616.71 | 1726.0 |
| distilgpt2 | 1 | 64 | fp32 | adamw | 1600.19 | 1738.0 |
| distilgpt2 | 2 | 64 | fp32 | adamw | 1616.21 | 1726.0 |
| distilgpt2 | 4 | 64 | fp32 | adamw | 1648.43 | 1768.0 |
| distilgpt2 | 1 | 64 | fp32 | sgd | 956.49 | 1030.0 |
| distilgpt2 | 1 | 64 | fp32 | adamw | 1600.94 | 1738.0 |
| gpt2 | 1 | 32 | fp32 | sgd | 1271.56 | 1388.0 |
| gpt2 | 1 | 64 | fp32 | sgd | 1279.20 | 1422.0 |
| gpt2 | 1 | 128 | fp32 | sgd | 1292.34 | 1406.0 |
| gpt2 | 1 | 32 | fp32 | adamw | 2402.82 | 2616.0 |
| gpt2 | 1 | 64 | fp32 | adamw | 2409.59 | 2670.0 |
| gpt2 | 1 | 128 | fp32 | adamw | 2422.73 | 2634.0 |

Total rows:

- distilgpt2 rows: 8
- gpt2 rows: 6
- combined rows: 14

## V3 Prediction Results

| model_name | optimizer | batch_size | input_tokens | actual_allocated | predicted_allocated | allocated_error_percent | actual_reserved | predicted_reserved | reserved_error_percent | backward_temp_factor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | adamw | 1 | 32 | 1592.43 | 1512.989686 | 4.988622 | 1700.0 | 1634.028860 | 3.880655 | 0.65 |
| distilgpt2 | adamw | 1 | 64 | 1600.69 | 1552.986061 | 2.980211 | 1738.0 | 1677.224945 | 3.496839 | 0.65 |
| distilgpt2 | adamw | 1 | 128 | 1616.71 | 1632.978811 | 1.006291 | 1726.0 | 1763.617115 | 2.179439 | 0.65 |
| distilgpt2 | adamw | 1 | 64 | 1600.19 | 1552.986061 | 2.949896 | 1738.0 | 1677.224945 | 3.496839 | 0.65 |
| distilgpt2 | adamw | 2 | 64 | 1616.21 | 1632.978811 | 1.037539 | 1726.0 | 1763.617115 | 2.179439 | 0.65 |
| distilgpt2 | adamw | 4 | 64 | 1648.43 | 1792.964311 | 8.767998 | 1768.0 | 1936.401455 | 9.524969 | 0.65 |
| distilgpt2 | sgd | 1 | 64 | 956.49 | 834.301197 | 12.774708 | 1030.0 | 901.045293 | 12.519874 | 0.35 |
| distilgpt2 | adamw | 1 | 64 | 1600.94 | 1552.986061 | 2.995361 | 1738.0 | 1677.224945 | 3.496839 | 0.65 |
| gpt2 | sgd | 1 | 32 | 1271.56 | 1196.307084 | 5.918157 | 1388.0 | 1292.011651 | 6.915587 | 0.35 |
| gpt2 | sgd | 1 | 64 | 1279.20 | 1257.068709 | 1.730088 | 1422.0 | 1357.634206 | 4.526427 | 0.35 |
| gpt2 | sgd | 1 | 128 | 1292.34 | 1378.591959 | 6.674092 | 1406.0 | 1488.879316 | 5.894688 | 0.35 |
| gpt2 | adamw | 1 | 32 | 2402.82 | 2288.117533 | 4.773660 | 2616.0 | 2471.166936 | 5.536432 | 0.65 |
| gpt2 | adamw | 1 | 64 | 2409.59 | 2348.879158 | 2.519551 | 2670.0 | 2536.789491 | 4.989158 | 0.65 |
| gpt2 | adamw | 1 | 128 | 2422.73 | 2470.402408 | 1.967714 | 2634.0 | 2668.034601 | 1.292126 | 0.65 |

## V3 Global Metrics

| estimator | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainingMemoryEstimatorV3 | 14 | 2.987786 | 4.363135 | 12.774708 | 1.006291 | 4.203541 | 4.994951 | 12.519874 | 1.292126 |

## V3 Error by Model

| model_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 8 | 4.687578 | 2.987786 | 12.774708 | 5.096862 | 3.496839 | 12.519874 | 1529.011250 | 1508.146375 | 1645.500000 | 1628.798084 |
| gpt2 | 6 | 3.930544 | 3.646606 | 6.674092 | 4.859070 | 5.262795 | 6.915587 | 1846.373333 | 1823.227809 | 2022.666667 | 1969.086033 |

## V3 Error by Optimizer

| optimizer_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 10 | 3.398684 | 2.965053 | 8.767998 | 4.007273 | 3.496839 | 9.524969 | 1851.0740 | 1833.826890 | 2005.4 | 1980.533041 |
| sgd | 4 | 6.774261 | 6.296124 | 12.774708 | 7.464144 | 6.405138 | 12.519874 | 1199.8975 | 1166.567237 | 1311.5 | 1259.892616 |

## V3 Error by Sequence Length

| model_name | input_tokens | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 32 | 1 | 4.988622 | 4.988622 | 4.988622 | 3.880655 | 3.880655 | 3.880655 | 1592.430 | 1512.989686 | 1700.0 | 1634.028860 |
| distilgpt2 | 64 | 6 | 5.250952 | 2.987786 | 12.774708 | 5.785800 | 3.496839 | 12.519874 | 1503.825 | 1486.533750 | 1623.0 | 1605.456450 |
| distilgpt2 | 128 | 1 | 1.006291 | 1.006291 | 1.006291 | 2.179439 | 2.179439 | 2.179439 | 1616.710 | 1632.978811 | 1726.0 | 1763.617115 |
| gpt2 | 32 | 2 | 5.345909 | 5.345909 | 5.918157 | 6.226010 | 6.226010 | 6.915587 | 1837.190 | 1742.212309 | 2002.0 | 1881.589293 |
| gpt2 | 64 | 2 | 2.124820 | 2.124820 | 2.519551 | 4.757792 | 4.757792 | 4.989158 | 1844.395 | 1802.973934 | 2046.0 | 1947.211848 |
| gpt2 | 128 | 2 | 4.320903 | 4.320903 | 6.674092 | 3.593407 | 3.593407 | 5.894688 | 1857.535 | 1924.497184 | 2020.0 | 2078.456958 |

## V2 vs V3 Metrics

| estimator | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainingMemoryEstimatorV2 | 14 | 2.987786 | 4.952768 | 17.693642 | 1.006291 | 3.688747 | 5.003712 | 16.833725 | 1.292126 |
| TrainingMemoryEstimatorV3 | 14 | 2.987786 | 4.363135 | 12.774708 | 1.006291 | 4.203541 | 4.994951 | 12.519874 | 1.292126 |

## V2 vs V3 Interpretation

V3 is not a clean replacement for V2.

V3 improved allocated mean error:

- V2 allocated mean error = 4.95%
- V3 allocated mean error = 4.36%

V3 improved max errors:

- allocated max error improved from 17.69% to 12.77%
- reserved max error improved from 16.83% to 12.52%

But V3 worsened reserved MRE:

- V2 reserved MRE = 3.69%
- V3 reserved MRE = 4.20%

So V3 is a mixed improvement.

## Important Row-wise Findings

V3 improved gpt2 SGD at longer sequence lengths.

For gpt2 SGD at 64 tokens:

- V2 allocated error = 9.40%
- V3 allocated error = 1.73%
- V2 reserved error = 6.29%
- V3 reserved error = 4.53%

For gpt2 SGD at 128 tokens:

- V2 allocated error = 17.69%
- V3 allocated error = 6.67%
- V2 reserved error = 16.83%
- V3 reserved error = 5.89%

This shows that reducing the SGD backward_temp_factor helped gpt2.

However, V3 damaged distilgpt2 SGD.

For distilgpt2 SGD at 64 tokens:

- V2 allocated error = 2.97%
- V3 allocated error = 12.77%
- V2 reserved error = 2.69%
- V3 reserved error = 12.52%

This is not acceptable for a final estimator.

## Main Interpretation

Day 34 showed that optimizer-specific correction alone is not enough.

The SGD factor of 0.35 improved gpt2 but hurt distilgpt2.

This means the correction should not depend only on optimizer name.

It likely needs to depend on model size or parameter scale as well.

Possible next correction:

- optimizer-specific plus model-size-aware backward temporary factor
- parameter-scale-aware SGD correction
- separate small-model and medium-model SGD factors

## Questions Answered

### Q1. Did V3 reduce gpt2 SGD error?

Yes.

The biggest improvement was gpt2 SGD at 128 tokens:

- allocated error improved from 17.69% to 6.67%
- reserved error improved from 16.83% to 5.89%

### Q2. Did V3 hurt AdamW error?

No.

AdamW rows stayed the same because AdamW backward_temp_factor remained 0.65.

### Q3. Did V3 improve global training error across both models?

Partially.

It improved allocated mean error and max errors, but worsened reserved MRE.

Allocated mean error:

- V2 = 4.95%
- V3 = 4.36%

Reserved MRE:

- V2 = 3.69%
- V3 = 4.20%

### Q4. Did V3 improve distilgpt2, gpt2, or both?

V3 improved gpt2.

It worsened distilgpt2 because of the distilgpt2 SGD row.

### Q5. Should final estimator use V2 or V3?

V2 should remain the main estimator for now.

V3 should be reported as an experimental correction.

The final estimator should not switch to V3 until the SGD correction becomes model-size-aware.

## Day 34 Conclusion

Day 34 created TrainingMemoryEstimator V3 with optimizer-specific backward temporary correction.

V3 improved gpt2 SGD prediction, especially at longer sequence lengths.

However, V3 damaged distilgpt2 SGD prediction.

Therefore, V3 is useful as an experimental correction but should not fully replace V2 yet.

The correct conclusion is that SGD correction needs to be optimizer-specific and model-size-aware.

The next step is Day 35: build a model-size-aware SGD/backward temporary correction.

---

# Day 35 - TrainingMemoryEstimator V4 Model-size-aware Correction

## Goal

Improve TrainingMemoryEstimator V3 by making SGD backward temporary correction model-size-aware.

Day 34 showed that V3 improved gpt2 SGD but damaged distilgpt2 SGD. This meant optimizer-specific correction alone was not enough.

Day 35 tested TrainingMemoryEstimator V4, which combines optimizer-specific correction with model-size-aware correction.

## Motivation

V2 used the same backward temporary factor for all optimizers:

- backward_temp_factor = 0.65

V3 changed SGD to:

- SGD backward_temp_factor = 0.35

This improved gpt2 SGD, but hurt distilgpt2 SGD.

V4 keeps the useful part of V3 for gpt2 while restoring V2 behavior for distilgpt2.

## V4 Rule

TrainingMemoryEstimator V4 uses optimizer-specific and model-size-aware backward temporary factors:

- AdamW: 0.65
- Adam: 0.65
- SGD for models below 100M parameters: 0.65
- SGD for models above/equal 100M parameters: 0.35
- SGD with momentum: 0.45
- default: 0.50

## Why This Rule Was Used

distilgpt2 has about 82M parameters, so its SGD factor remains 0.65.

gpt2 has about 124M parameters, so its SGD factor becomes 0.35.

This should preserve V2 behavior on distilgpt2 and V3 behavior on gpt2.

## Files Created

- src/estimators/training_memory_estimator_v4.py
- results/training_estimator_v4_predictions.csv
- results/training_estimator_v4_metrics.csv
- results/training_estimator_v4_error_by_model.csv
- results/training_estimator_v4_error_by_optimizer.csv
- results/training_estimator_v4_error_by_sequence_length.csv
- results/training_estimator_v2_v3_v4_metrics.csv
- results/training_estimator_v2_v3_v4_row_comparison.csv
- plots/training_estimator_v2_v3_v4_allocated_mre.png
- plots/training_estimator_v2_v3_v4_reserved_mre.png
- plots/training_estimator_v4_actual_vs_predicted_allocated.png
- plots/training_estimator_v4_actual_vs_predicted_reserved.png
- report/training_estimator_v4_model_size_correction_report.md

## V4 Prediction Results

| model_name | batch_size | input_tokens | optimizer | actual_allocated | predicted_allocated | allocated_error_percent | actual_reserved | predicted_reserved | reserved_error_percent | num_parameters | backward_temp_factor |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 1 | 32 | adamw | 1592.43 | 1512.989686 | 4.988622 | 1700.0 | 1634.028860 | 3.880655 | 81912576 | 0.65 |
| distilgpt2 | 1 | 64 | adamw | 1600.69 | 1552.986061 | 2.980211 | 1738.0 | 1677.224945 | 3.496839 | 81912576 | 0.65 |
| distilgpt2 | 1 | 128 | adamw | 1616.71 | 1632.978811 | 1.006291 | 1726.0 | 1763.617115 | 2.179439 | 81912576 | 0.65 |
| distilgpt2 | 1 | 64 | adamw | 1600.19 | 1552.986061 | 2.949896 | 1738.0 | 1677.224945 | 3.496839 | 81912576 | 0.65 |
| distilgpt2 | 2 | 64 | adamw | 1616.21 | 1632.978811 | 1.037539 | 1726.0 | 1763.617115 | 2.179439 | 81912576 | 0.65 |
| distilgpt2 | 4 | 64 | adamw | 1648.43 | 1792.964311 | 8.767998 | 1768.0 | 1936.401455 | 9.524969 | 81912576 | 0.65 |
| distilgpt2 | 1 | 64 | sgd | 956.49 | 928.042701 | 2.974134 | 1030.0 | 1002.286117 | 2.690668 | 81912576 | 0.65 |
| distilgpt2 | 1 | 64 | adamw | 1600.94 | 1552.986061 | 2.995361 | 1738.0 | 1677.224945 | 3.496839 | 81912576 | 0.65 |
| gpt2 | 1 | 32 | sgd | 1271.56 | 1196.307084 | 5.918157 | 1388.0 | 1292.011651 | 6.915587 | 124439808 | 0.35 |
| gpt2 | 1 | 64 | sgd | 1279.20 | 1257.068709 | 1.730088 | 1422.0 | 1357.634206 | 4.526427 | 124439808 | 0.35 |
| gpt2 | 1 | 128 | sgd | 1292.34 | 1378.591959 | 6.674092 | 1406.0 | 1488.879316 | 5.894688 | 124439808 | 0.35 |
| gpt2 | 1 | 32 | adamw | 2402.82 | 2288.117533 | 4.773660 | 2616.0 | 2471.166936 | 5.536432 | 124439808 | 0.65 |
| gpt2 | 1 | 64 | adamw | 2409.59 | 2348.879158 | 2.519551 | 2670.0 | 2536.789491 | 4.989158 | 124439808 | 0.65 |
| gpt2 | 1 | 128 | adamw | 2422.73 | 2470.402408 | 1.967714 | 2634.0 | 2668.034601 | 1.292126 | 124439808 | 0.65 |

## V4 Global Metrics

| estimator | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainingMemoryEstimatorV4 | 14 | 2.977173 | 3.663094 | 8.767998 | 1.006291 | 3.688747 | 4.292865 | 9.524969 | 1.292126 |

## V4 Error by Model

| model_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 8 | 3.462507 | 2.977173 | 8.767998 | 3.868211 | 3.496839 | 9.524969 | 1529.011250 | 1519.864062 | 1645.500000 | 1641.453187 |
| gpt2 | 6 | 3.930544 | 3.646606 | 6.674092 | 4.859070 | 5.262795 | 6.915587 | 1846.373333 | 1823.227809 | 2022.666667 | 1969.086033 |

## Model-wise Interpretation

V4 preserved strong distilgpt2 performance while improving gpt2 training prediction.

For distilgpt2:

- allocated mean error = 3.46%
- reserved mean error = 3.87%

For gpt2:

- allocated mean error = 3.93%
- reserved mean error = 4.86%

This is a much more balanced result than V2 or V3.

## V4 Error by Optimizer

| optimizer_name | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 10 | 3.398684 | 2.965053 | 8.767998 | 4.007273 | 3.496839 | 9.524969 | 1851.0740 | 1833.826890 | 2005.4 | 1980.533041 |
| sgd | 4 | 4.324118 | 4.446146 | 6.674092 | 5.006843 | 5.210558 | 6.915587 | 1199.8975 | 1190.002613 | 1311.5 | 1285.202822 |

## Optimizer-wise Interpretation

V4 made optimizer-wise error more balanced.

AdamW:

- allocated mean error = 3.40%
- reserved mean error = 4.01%

SGD:

- allocated mean error = 4.32%
- reserved mean error = 5.01%

This is acceptable across both optimizers.

## V4 Error by Sequence Length

| model_name | input_tokens | num_rows | allocated_mean_error | allocated_median_error | allocated_max_error | reserved_mean_error | reserved_median_error | reserved_max_error | avg_actual_allocated_MB | avg_predicted_allocated_MB | avg_actual_reserved_MB | avg_predicted_reserved_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 32 | 1 | 4.988622 | 4.988622 | 4.988622 | 3.880655 | 3.880655 | 3.880655 | 1592.430 | 1512.989686 | 1700.0 | 1634.028860 |
| distilgpt2 | 64 | 6 | 3.617523 | 2.977173 | 8.767998 | 4.147599 | 3.496839 | 9.524969 | 1503.825 | 1502.157334 | 1623.0 | 1622.329921 |
| distilgpt2 | 128 | 1 | 1.006291 | 1.006291 | 1.006291 | 2.179439 | 2.179439 | 2.179439 | 1616.710 | 1632.978811 | 1726.0 | 1763.617115 |
| gpt2 | 32 | 2 | 5.345909 | 5.345909 | 5.918157 | 6.226010 | 6.226010 | 6.915587 | 1837.190 | 1742.212309 | 2002.0 | 1881.589293 |
| gpt2 | 64 | 2 | 2.124820 | 2.124820 | 2.519551 | 4.757792 | 4.757792 | 4.989158 | 1844.395 | 1802.973934 | 2046.0 | 1947.211848 |
| gpt2 | 128 | 2 | 4.320903 | 4.320903 | 6.674092 | 3.593407 | 3.593407 | 5.894688 | 1857.535 | 1924.497184 | 2020.0 | 2078.456958 |

## V2 vs V3 vs V4 Metrics

| estimator | num_rows | allocated_MRE | allocated_mean_error | allocated_max_error | allocated_min_error | reserved_MRE | reserved_mean_error | reserved_max_error | reserved_min_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainingMemoryEstimatorV2 | 14 | 2.987786 | 4.952768 | 17.693642 | 1.006291 | 3.688747 | 5.003712 | 16.833725 | 1.292126 |
| TrainingMemoryEstimatorV3 | 14 | 2.987786 | 4.363135 | 12.774708 | 1.006291 | 4.203541 | 4.994951 | 12.519874 | 1.292126 |
| TrainingMemoryEstimatorV4 | 14 | 2.977173 | 3.663094 | 8.767998 | 1.006291 | 3.688747 | 4.292865 | 9.524969 | 1.292126 |

## V2 vs V3 vs V4 Interpretation

V4 is the strongest estimator so far.

Compared with V2:

- allocated mean error improved from 4.95% to 3.66%
- allocated max error improved from 17.69% to 8.77%
- reserved mean error improved from 5.00% to 4.29%
- reserved max error improved from 16.83% to 9.52%

Compared with V3:

- V4 fixed the distilgpt2 SGD damage
- V4 kept the gpt2 SGD improvement
- V4 reduced allocated mean error further
- V4 restored reserved MRE to V2 level

## Targeted SGD Rows

| model_name | input_tokens | optimizer | actual_allocated | V2_allocated_error | V3_allocated_error | V4_allocated_error | actual_reserved | V2_reserved_error | V3_reserved_error | V4_reserved_error | V3_factor | V4_factor |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 64 | sgd | 956.49 | 2.974134 | 12.774708 | 2.974134 | 1030.0 | 2.690668 | 12.519874 | 2.690668 | 0.35 | 0.65 |
| gpt2 | 32 | sgd | 1271.56 | 5.281476 | 5.918157 | 5.918157 | 1388.0 | 4.165311 | 6.915587 | 6.915587 | 0.35 | 0.35 |
| gpt2 | 64 | sgd | 1279.20 | 9.402655 | 1.730088 | 1.730088 | 1422.0 | 6.289527 | 4.526427 | 4.526427 | 0.35 | 0.35 |
| gpt2 | 128 | sgd | 1292.34 | 17.693642 | 6.674092 | 6.674092 | 1406.0 | 16.833725 | 5.894688 | 5.894688 | 0.35 | 0.35 |

## Targeted SGD Interpretation

V4 fixed the distilgpt2 SGD damage from V3.

For distilgpt2 SGD at 64 tokens:

- V2 allocated error = 2.97%
- V3 allocated error = 12.77%
- V4 allocated error = 2.97%

Reserved error:

- V2 reserved error = 2.69%
- V3 reserved error = 12.52%
- V4 reserved error = 2.69%

So V4 fully restored the correct behavior for distilgpt2 SGD.

V4 also preserved the gpt2 SGD improvement from V3.

For gpt2 SGD at 128 tokens:

- V2 allocated error = 17.69%
- V3 allocated error = 6.67%
- V4 allocated error = 6.67%

Reserved error:

- V2 reserved error = 16.83%
- V3 reserved error = 5.89%
- V4 reserved error = 5.89%

So V4 kept the improvement on gpt2 while fixing the damage on distilgpt2.

## Questions Answered

### Q1. Did V4 fix the distilgpt2 SGD damage from V3?

Yes.

For distilgpt2 SGD at 64 tokens, V4 restored allocated error from 12.77% back to 2.97%. Reserved error also returned from 12.52% back to 2.69%.

### Q2. Did V4 preserve the gpt2 SGD improvement from V3?

Yes.

For gpt2 SGD at 128 tokens, V4 kept the improved allocated error of 6.67%, compared with V2's 17.69%. Reserved error stayed at 5.89%, compared with V2's 16.83%.

### Q3. Did V4 improve global metrics compared with V2 and V3?

Yes.

V4 gave the best global metrics so far.

Allocated mean error:

- V2 = 4.95%
- V3 = 4.36%
- V4 = 3.66%

Reserved mean error:

- V2 = 5.00%
- V3 = 4.99%
- V4 = 4.29%

### Q4. Did V4 hurt AdamW?

No.

AdamW stayed unchanged because the AdamW backward_temp_factor remained 0.65.

### Q5. Should V4 become the final training estimator?

Yes, based on the current validation set, V4 should become the candidate final training estimator.

The correct wording is that TrainingMemoryEstimator V4 is the strongest estimator on the current validation set and is the current candidate final training estimator.

## Day 35 Conclusion

Day 35 built TrainingMemoryEstimator V4 using optimizer-specific and model-size-aware backward temporary correction.

V4 solved the exact failure pattern discovered on Day 34.

It fixed the distilgpt2 SGD damage caused by V3 while preserving the gpt2 SGD improvement.

V4 also improved global metrics compared with both V2 and V3.

V4 reduced allocated mean error to 3.66% and reserved mean error to 4.29% across the combined distilgpt2 + gpt2 training validation set.

Therefore, TrainingMemoryEstimator V4 is the strongest training estimator so far and should be treated as the candidate final training estimator.

The next step is Day 36: final training estimator report and clean GitHub update.

---

# Day 36 - Final V4 Training Estimator Report

## Goal

Create a clean final report for the TrainingMemoryEstimator V4 module.

## Important Clarification

Day 36 is not the final project report.

It only finalizes the current V4 training estimator work before moving to PEF, quantization, sparsity, model-parallelism, and architecture comparison.

## Files Created

- report/final_training_estimator_report.md
- report/readme_v4_training_estimator_section.md
- results/final_training_estimator_metrics.csv
- results/final_training_estimator_key_findings.csv
- results/final_training_estimator_comparison.csv

## Final V4 Metrics

              metric  value_percent                                             meaning
       allocated_MRE       2.977173     Median relative error for peak allocated memory
allocated_mean_error       3.663094       Mean relative error for peak allocated memory
 allocated_max_error       8.767998 Worst-case relative error for peak allocated memory
        reserved_MRE       3.688747      Median relative error for peak reserved memory
 reserved_mean_error       4.292865        Mean relative error for peak reserved memory
  reserved_max_error       9.524969  Worst-case relative error for peak reserved memory

## Final Estimator Comparison

                estimator                                                                                  role  num_rows  allocated_MRE  allocated_mean_error  allocated_max_error  reserved_MRE  reserved_mean_error  reserved_max_error
TrainingMemoryEstimatorV2             V2: baseline training estimator with backward temporary memory correction        14       2.987786              4.952768            17.693642      3.688747             5.003712           16.833725
TrainingMemoryEstimatorV3                                          V3: optimizer-specific correction experiment        14       2.987786              4.363135            12.774708      4.203541             4.994951           12.519874
TrainingMemoryEstimatorV4 V4: candidate final estimator with optimizer-specific and model-size-aware correction        14       2.977173              3.663094             8.767998      3.688747             4.292865            9.524969

## Key Findings

 finding_id                                                                 finding                                                                                                                                                              evidence                                                                    importance
          1 TrainingMemoryEstimator V4 is the strongest current training estimator. V4 achieved 2.98% allocated MRE, 3.66% allocated mean error, 3.69% reserved MRE, and 4.29% reserved mean error across combined distilgpt2 + gpt2 training validation.    This makes V4 the candidate final estimator for the training-memory phase.
          2                       V4 fixed the distilgpt2 SGD failure caused by V3.                                                                        For distilgpt2 SGD at 64 tokens, V3 allocated error was 12.77%, while V4 restored it to 2.97%.                  This showed that optimizer-only correction was insufficient.
          3                          V4 preserved the gpt2 SGD improvement from V3.                                                                                 For gpt2 SGD at 128 tokens, V2 allocated error was 17.69%, while V4 kept it at 6.67%.         This improved training-memory prediction for larger GPT-style models.
          4                 Model-size-aware correction improved estimator balance.                                                                                   V4 uses SGD factor 0.65 below 100M parameters and 0.35 above/equal 100M parameters.                                   This better handles different model scales.
          5                        V4 reduced worst-case training prediction error.                                                        Allocated max error dropped from 17.69% in V2 to 8.77% in V4; reserved max error dropped from 16.83% to 9.52%. Lower worst-case error is important for deployment-style fit/fail prediction.

## Main Interpretation

TrainingMemoryEstimator V4 is the strongest estimator on the current combined distilgpt2 + gpt2 training validation set.

V4 fixed the distilgpt2 SGD damage caused by V3 while preserving the gpt2 SGD improvement.

V4 should be treated as the candidate final training estimator for now.

## Next Step

Day 37 will run V4 training PEF-style fit/fail simulation.

---

# Day 37 - V4 Training PEF-style Fit/Fail Simulation

## Goal

Evaluate whether TrainingMemoryEstimator V4 can correctly predict fit/fail decisions under different GPU memory limits.

## Why This Matters

MRE tells how close the predicted memory is numerically.

PEF-style fit/fail simulation checks whether the estimator is useful for deployment-style GPU placement decisions.

A low-error estimator can still make wrong fit/fail decisions near memory boundaries.

## Method

For each training workload and memory limit:

- actual_fits = actual_peak_reserved_MB <= memory_limit_MB
- predicted_fits = predicted_peak_reserved_MB <= memory_limit_MB

Each case was classified as:

- correct
- dangerous_underprediction
- conservative_overprediction

Dangerous underprediction means the estimator predicts that a workload fits, but it actually does not fit.

Conservative overprediction means the estimator predicts that a workload does not fit, but it actually fits.

## Files Created

- results/v4_training_pef_cases.csv
- results/v4_training_pef_summary.csv
- results/v4_training_pef_by_memory_limit.csv
- results/v4_training_pef_by_model.csv
- results/v4_training_pef_by_optimizer.csv
- results/v4_training_pef_dangerous_failures.csv
- results/v4_training_pef_conservative_failures.csv
- plots/v4_training_pef_accuracy_by_memory_limit.png
- plots/v4_training_pef_failure_rate_by_memory_limit.png
- plots/v4_training_pef_accuracy_by_model.png
- plots/v4_training_pef_accuracy_by_optimizer.png
- report/v4_training_pef_simulation_report.md

## Global PEF Summary

| total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|
| 266 | 256 | 8 | 2 | 96.240602 | 3.007519 | 0.751880 |

## PEF by Memory Limit

| memory_limit_MB | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 1024 | 14 | 13 | 1 | 0 | 92.857143 | 7.142857 | 0.000000 |
| 1200 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 1300 | 14 | 13 | 1 | 0 | 92.857143 | 7.142857 | 0.000000 |
| 1400 | 14 | 13 | 1 | 0 | 92.857143 | 7.142857 | 0.000000 |
| 1500 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 1600 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 1700 | 14 | 11 | 3 | 0 | 78.571429 | 21.428571 | 0.000000 |
| 1800 | 14 | 13 | 0 | 1 | 92.857143 | 0.000000 | 7.142857 |
| 1900 | 14 | 13 | 0 | 1 | 92.857143 | 0.000000 | 7.142857 |
| 2000 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 2048 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 2200 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 2400 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 2600 | 14 | 12 | 2 | 0 | 85.714286 | 14.285714 | 0.000000 |
| 2800 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 3072 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 4096 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |
| 8192 | 14 | 14 | 0 | 0 | 100.000000 | 0.000000 | 0.000000 |

## PEF by Model

| model_name | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilgpt2 | 152 | 146 | 4 | 2 | 96.052632 | 2.631579 | 1.315789 |
| gpt2 | 114 | 110 | 4 | 0 | 96.491228 | 3.508772 | 0.000000 |

## PEF by Optimizer

| optimizer_name | total_cases | correct_cases | dangerous_failures | conservative_failures | accuracy_percent | dangerous_failure_rate_percent | conservative_failure_rate_percent |
|---|---:|---:|---:|---:|---:|---:|---:|
| adamw | 190 | 183 | 5 | 2 | 96.315789 | 2.631579 | 1.052632 |
| sgd | 76 | 73 | 3 | 0 | 96.052632 | 3.947368 | 0.000000 |

## Dangerous Failures

| model_name | batch_size | input_tokens | optimizer_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB | outcome |
|---|---:|---:|---|---:|---:|---:|---|
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | dangerous_underprediction |
| distilgpt2 | 1 | 64 | sgd | 1024 | 1030.0 | 1002.286117 | dangerous_underprediction |
| distilgpt2 | 1 | 64 | adamw | 1700 | 1738.0 | 1677.224945 | dangerous_underprediction |
| gpt2 | 1 | 32 | sgd | 1300 | 1388.0 | 1292.011651 | dangerous_underprediction |
| gpt2 | 1 | 64 | sgd | 1400 | 1422.0 | 1357.634206 | dangerous_underprediction |
| gpt2 | 1 | 32 | adamw | 2600 | 2616.0 | 2471.166936 | dangerous_underprediction |
| gpt2 | 1 | 64 | adamw | 2600 | 2670.0 | 2536.789491 | dangerous_underprediction |

## Conservative Failures

| model_name | batch_size | input_tokens | optimizer_name | memory_limit_MB | actual_peak_reserved_MB | predicted_peak_reserved_MB | outcome |
|---|---:|---:|---|---:|---:|---:|---|
| distilgpt2 | 4 | 64 | adamw | 1800 | 1768.0 | 1936.401455 | conservative_overprediction |
| distilgpt2 | 4 | 64 | adamw | 1900 | 1768.0 | 1936.401455 | conservative_overprediction |

## Main Findings

V4 achieved 96.24% PEF-style fit/fail accuracy across 266 simulated training placement cases.

This is better than the earlier training PEF result of 91.07%, showing that V4 improved deployment-style fit/fail prediction, not just MRE.

However, 8 dangerous underpredictions still remained.

Most failures occurred near tight memory boundaries, especially at 1700 MB and 2600 MB.

At relaxed memory limits such as 2048 MB, 3072 MB, 4096 MB, and 8192 MB, predictions were fully correct.

Model-wise, distilgpt2 and gpt2 performed similarly.

Optimizer-wise, AdamW and SGD also had similar accuracy, though SGD had a slightly higher dangerous failure rate.

## Main Interpretation

V4 is useful for fit/fail decisions, but not fully safe without a margin.

The remaining dangerous failures occur because predicted reserved memory is close to the memory limit but slightly below it, while actual reserved memory is slightly above it.

This shows why PEF-style evaluation is stricter than MRE.

## Next Step

Day 38 will run safety-margin simulation for V4 training predictions.

The goal is to remove dangerous underpredictions while keeping conservative failures reasonable.

---

# Day 38 - V4 Training PEF Safety-margin Simulation

## Goal

Test whether adding safety margin to V4 predicted reserved memory can remove dangerous underpredictions.

## Context

Day 37 showed V4 achieved 96.24% PEF-style accuracy but still had 8 dangerous failures.

Day 38 tests margins of 0%, 2%, 5%, 7.5%, 10%, and 15%.

## Files Created

- results/v4_training_pef_safety_margin_cases.csv
- results/v4_training_pef_safety_margin_summary.csv
- results/v4_training_pef_safety_margin_by_limit.csv
- results/v4_training_pef_best_safety_margin.csv
- results/v4_training_pef_safety_margin_failures.csv
- plots/v4_training_pef_accuracy_vs_safety_margin.png
- plots/v4_training_pef_dangerous_failure_vs_safety_margin.png
- plots/v4_training_pef_conservative_failure_vs_safety_margin.png
- plots/v4_training_pef_tradeoff_vs_safety_margin.png
- report/v4_training_pef_safety_margin_report.md

## Safety-margin Summary

 safety_margin  safety_margin_percent  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
         0.000                    0.0          266            256                   8                      2         96.240602                        3.007519                           0.751880
         0.020                    2.0          266            259                   4                      3         97.368421                        1.503759                           1.127820
         0.050                    5.0          266            257                   1                      8         96.616541                        0.375940                           3.007519
         0.075                    7.5          266            253                   0                     13         95.112782                        0.000000                           4.887218
         0.100                   10.0          266            250                   0                     16         93.984962                        0.000000                           6.015038
         0.150                   15.0          266            239                   0                     27         89.849624                        0.000000                          10.150376

## Best Safety Margin

 safety_margin  safety_margin_percent  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
         0.075                    7.5          266            253                   0                     13         95.112782                             0.0                           4.887218

## Failures at Best Margin

 safety_margin  safety_margin_percent model_name  batch_size  input_tokens dtype optimizer_name  memory_limit_MB  actual_peak_allocated_MB  predicted_peak_allocated_MB  actual_peak_reserved_MB  predicted_peak_reserved_MB  safe_predicted_reserved_MB  actual_fits  predicted_fits  correct_prediction  dangerous_failure  conservative_failure                     outcome
         0.075                    7.5 distilgpt2           1            32  fp32          adamw             1700                   1592.43                  1512.989686                   1700.0                 1634.028860                 1756.581025         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           1            64  fp32          adamw             1800                   1600.69                  1552.986061                   1738.0                 1677.224945                 1803.016816         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           1           128  fp32          adamw             1800                   1616.71                  1632.978811                   1726.0                 1763.617115                 1895.888399         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           1            64  fp32          adamw             1800                   1600.19                  1552.986061                   1738.0                 1677.224945                 1803.016816         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           2            64  fp32          adamw             1800                   1616.21                  1632.978811                   1726.0                 1763.617115                 1895.888399         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           4            64  fp32          adamw             1800                   1648.43                  1792.964311                   1768.0                 1936.401455                 2081.631565         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           4            64  fp32          adamw             1900                   1648.43                  1792.964311                   1768.0                 1936.401455                 2081.631565         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           4            64  fp32          adamw             2000                   1648.43                  1792.964311                   1768.0                 1936.401455                 2081.631565         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           4            64  fp32          adamw             2048                   1648.43                  1792.964311                   1768.0                 1936.401455                 2081.631565         True           False               False              False                  True conservative_overprediction
         0.075                    7.5 distilgpt2           1            64  fp32          adamw             1800                   1600.94                  1552.986061                   1738.0                 1677.224945                 1803.016816         True           False               False              False                  True conservative_overprediction
         0.075                    7.5       gpt2           1           128  fp32            sgd             1500                   1292.34                  1378.591959                   1406.0                 1488.879316                 1600.545264         True           False               False              False                  True conservative_overprediction
         0.075                    7.5       gpt2           1           128  fp32            sgd             1600                   1292.34                  1378.591959                   1406.0                 1488.879316                 1600.545264         True           False               False              False                  True conservative_overprediction
         0.075                    7.5       gpt2           1           128  fp32          adamw             2800                   2422.73                  2470.402408                   2634.0                 2668.034601                 2868.137196         True           False               False              False                  True conservative_overprediction

## Main Interpretation

Day 38 tested the safety tradeoff between dangerous underprediction and conservative overprediction.

A safety margin is needed because V4 still had dangerous failures near tight memory boundaries.

## Next Step

Day 39 will compare inference PEF and training PEF results.

---

# Day 39 - Inference vs Training PEF Comparison

## Goal

Compare inference PEF and training PEF results to understand fit/fail prediction reliability across phases.

## Files Created

- results/inference_vs_training_pef_comparison.csv
- results/inference_vs_training_safety_comparison.csv
- results/inference_vs_training_key_findings.csv
- plots/inference_vs_training_pef_accuracy.png
- plots/inference_vs_training_dangerous_failure_rate.png
- plots/inference_vs_training_conservative_failure_rate.png
- report/inference_vs_training_pef_comparison_report.md

## Main Comparison

    phase                                 estimator  safety_margin_percent  total_cases  correct_cases  dangerous_failures  conservative_failures  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent
inference                CombinedInferenceEstimator                    0.0        720.0          678.0                40.0                    2.0         94.170000                        5.560000                           0.280000
 training                 TrainingMemoryEstimatorV4                    0.0        266.0          256.0                 8.0                    2.0         96.240602                        3.007519                           0.751880
 training TrainingMemoryEstimatorV4 + safety margin                    7.5        266.0          253.0                 0.0                   13.0         95.112782                        0.000000                           4.887218

## Safety Comparison

             comparison     phase  safety_margin_percent  accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent                                                                      interpretation
    inference_no_margin inference                    0.0         94.170000                        5.560000                           0.280000                                                 Baseline inference PEF performance.
  training_v4_no_margin  training                    0.0         96.240602                        3.007519                           0.751880                 V4 improves training PEF accuracy but still has dangerous failures.
training_v4_safe_margin  training                    7.5         95.112782                        0.000000                           4.887218 Recommended safe training placement setting because dangerous failures are removed.

## Key Findings

 finding_id                                                                 finding                                                                                      evidence                                                                            interpretation
          1 Training V4 PEF accuracy is higher than overall inference PEF accuracy.        Training V4 achieved 96.24% accuracy, while overall inference PEF accuracy was 94.17%.             V4 improved training fit/fail prediction despite training being more complex.
          2            Training V4 still needs safety margin for deployment safety. Without margin, V4 had 8 dangerous failures. At 7.5% margin, dangerous failures dropped to 0.                                  Raw accuracy alone is not enough for safe GPU placement.
          3                    The 7.5% training margin trades accuracy for safety.           Accuracy dropped from 96.24% to 95.11%, but dangerous failures dropped from 8 to 0.            This is acceptable when avoiding OOM matters more than maximizing utilization.
          4                                               PEF is stricter than MRE.           Small reserved-memory errors near memory limits caused dangerous fit/fail mistakes.          A numerically accurate estimator can still be unsafe near scheduling boundaries.
          5          Reserved memory is the correct basis for placement simulation.   PEF simulation used actual and predicted peak reserved memory rather than allocated memory. CUDA allocator reservation determines whether a workload may hit memory-limit boundaries.

## Main Interpretation

Day 39 showed that PEF-style evaluation is stricter than MRE because small memory errors near boundaries can flip placement decisions.

TrainingMemoryEstimator V4 achieved strong fit/fail accuracy, but safety margin is required to remove dangerous underpredictions.

V4 with 7.5% safety margin is the recommended safe training placement setting.

## Next Step

Day 40 will create a final estimator comparison table across inference and training estimators.

---

# Day 40 - Final Estimator Comparison Table

## Goal

Create a consolidated estimator comparison across inference, training, PEF-style fit/fail simulation, and safety-margin behavior.

## Files Created

- results/final_estimator_comparison_table.csv
- results/final_phase_summary.csv
- results/final_recommended_estimators.csv
- results/final_estimator_key_findings.csv
- plots/final_estimator_allocated_mean_error.png
- plots/final_estimator_reserved_mean_error.png
- plots/final_pef_accuracy_comparison.png
- plots/final_dangerous_failure_comparison.png
- report/final_estimator_comparison_report.md

## Final Estimator Comparison

    phase                                  estimator           validation_scope  allocated_MRE_percent  allocated_mean_error_percent  allocated_max_error_percent  reserved_MRE_percent  reserved_mean_error_percent  reserved_max_error_percent  pef_accuracy_percent  dangerous_failure_rate_percent  conservative_failure_rate_percent  safety_margin_percent                                                             status
inference                 CombinedInferenceEstimator                 distilgpt2                    NaN                          2.36                          NaN                   NaN                         2.36                         NaN                 99.52                            0.00                                NaN                    0.0                              strong inference result on distilgpt2
inference                 CombinedInferenceEstimator                       gpt2                   2.66                          2.50                         5.09                  2.06                         2.28                        5.22                   NaN                             NaN                                NaN                    0.0       validated inference generalization to larger GPT-style model
 training                 TrainingMemoryEstimator V1                 distilgpt2                  15.66                         14.45                        24.21                 16.12                        14.13                       23.99                   NaN                             NaN                                NaN                    0.0  baseline; underpredicted due to missing backward temporary memory
 training                 TrainingMemoryEstimator V2 distilgpt2 + gpt2 combined                   2.99                          4.95                        17.69                  3.69                         5.00                       16.83                   NaN                             NaN                                NaN                    0.0              good baseline training estimator but weak on gpt2 SGD
 training                 TrainingMemoryEstimator V3 distilgpt2 + gpt2 combined                   2.99                          4.36                        12.77                  4.20                         4.99                       12.52                   NaN                             NaN                                NaN                    0.0 experimental optimizer-specific correction; damaged distilgpt2 SGD
 training                 TrainingMemoryEstimator V4 distilgpt2 + gpt2 combined                   2.98                          3.66                         8.77                  3.69                         4.29                        9.52                 96.24                            3.01                               0.75                    0.0              candidate final training estimator for raw prediction
 training TrainingMemoryEstimator V4 + safety margin distilgpt2 + gpt2 combined                   2.98                          3.66                         8.77                  3.69                         4.29                        9.52                 95.11                            0.00                               4.89                    7.5                    recommended safe setting for training placement

## Phase Summary

    phase best_estimator_for_prediction                                                            best_validation_result                                                 best_pef_result                                                                             main_memory_components                                                                                                    main_conclusion
inference    CombinedInferenceEstimator             gpt2 inference: 2.50% allocated mean error, 2.28% reserved mean error distilgpt2 inference: 99.52% PEF accuracy, 0 dangerous failures                                       parameters, activations, KV cache, dtype, allocator behavior  Inference memory can be predicted accurately using modular parameter/token/cache/precision/allocator corrections.
 training    TrainingMemoryEstimator V4 combined distilgpt2 + gpt2: 3.66% allocated mean error, 4.29% reserved mean error  V4 + 7.5% safety margin: 95.11% accuracy, 0 dangerous failures parameters, gradients, optimizer states, activations, backward temporary memory, allocator padding Training memory needs separate modeling because optimizer state and backward temporary memory dominate peak usage.

## Recommended Estimators

                          use_case                           recommended_estimator                                                                   reason                                               important_metric                                                                     caution
   LLM inference memory prediction                      CombinedInferenceEstimator Strong low-error prediction on distilgpt2 and gpt2 inference validation. gpt2 allocated mean error = 2.50%, reserved mean error = 2.28% Tiny-gpt2 is too small for realistic evaluation because overhead dominates.
LLM training raw memory prediction                      TrainingMemoryEstimator V4             Best combined training estimator across distilgpt2 and gpt2.      allocated mean error = 3.66%, reserved mean error = 4.29%      100M parameter threshold is empirical and needs more model validation.
  safe training workload placement TrainingMemoryEstimator V4 + 7.5% safety margin            Removes all dangerous underpredictions in current validation.                      0 dangerous failures, 95.11% PEF accuracy           Increases conservative failures, so GPU utilization may decrease.

## Key Findings

 finding_id                                                          finding                                                                                                       evidence                                                                                    why_it_matters
          1 Inference memory prediction generalizes from distilgpt2 to gpt2.                              gpt2 inference achieved 2.50% allocated mean error and 2.28% reserved mean error.                                                 Shows estimator is not only fitted to distilgpt2.
          2   Training memory needs separate modeling from inference memory.                                  Training used around 4.7x to 4.8x more memory than comparable inference runs. Gradients, optimizer states, and backward temporary memory make training fundamentally different.
          3           AdamW optimizer state is close to 2x parameter memory.           distilgpt2 estimate 624.94 MB vs observed 644.45 MB; gpt2 estimate 949.40 MB vs observed ~950.92 MB.                             Optimizer-state modeling is essential for training memory prediction.
          4                  V4 is the strongest current training estimator. V4 achieved 3.66% allocated mean error, 4.29% reserved mean error, and reduced max errors compared with V2/V3.                                                V4 becomes the candidate final training estimator.
          5                       PEF-style evaluation is stricter than MRE.                               V4 had low error but still produced dangerous failures near tight memory limits.              Deployment decisions require fit/fail evaluation, not only numeric prediction error.
          6       V4 needs a 7.5% safety margin for safe training placement.                                              7.5% margin reduced dangerous failures to 0 with 95.11% accuracy.                            Safe scheduling prioritizes avoiding OOM over maximizing raw accuracy.

## Main Interpretation

Day 40 created the main comparison table for the project so far.

CombinedInferenceEstimator is the recommended inference estimator.

TrainingMemoryEstimator V4 is the recommended raw training memory estimator.

TrainingMemoryEstimator V4 with 7.5% safety margin is the recommended safe training placement setting.

## Next Step

Day 41 will start the precision and quantization adaptation phase.

---

# Day 41 - Precision Memory Adaptation: fp32 vs fp16

## Goal

Create a formal precision-memory comparison for fp32 vs fp16 inference workloads.

## Context

Day 40 closed the core estimator comparison phase.

Day 41 starts the precision and quantization adaptation phase.

## Files Created

- results/precision_memory_comparison.csv
- results/precision_memory_key_findings.csv
- results/precision_reduction_summary.csv
- plots/precision_peak_allocated_comparison.png
- plots/precision_peak_reserved_comparison.png
- plots/precision_memory_reduction_percent.png
- report/precision_memory_adaptation_report.md

## Precision Comparison

model_name      task  batch_size  input_tokens  max_new_tokens  fp32_peak_allocated_MB  fp16_peak_allocated_MB  fp32_peak_reserved_MB  fp16_peak_reserved_MB                                         source_note  allocated_reduction_MB  allocated_reduction_percent  reserved_reduction_MB  reserved_reduction_percent
distilgpt2 inference           1            64             128                  343.41                  181.25                    NaN                    NaN distilgpt2 fp32/fp16 inference precision experiment                  162.16                    47.220524                    NaN                         NaN
      gpt2 inference           1           128             128                  504.11                  264.78                  558.0                  268.0                 gpt2 fp32/fp16 inference validation                  239.33                    47.475749                  290.0                   51.971326

## Reduction Summary

                             metric     value                                                                                         interpretation
average_allocated_reduction_percent 47.348136 Average peak allocated memory reduction from fp32 to fp16 across tested GPT-style inference workloads.
    min_allocated_reduction_percent 47.220524                                        Smallest observed allocated memory reduction from fp32 to fp16.
    max_allocated_reduction_percent 47.475749                                         Largest observed allocated memory reduction from fp32 to fp16.
    gpt2_reserved_reduction_percent 51.971326                                    Observed reserved memory reduction for gpt2 fp32 to fp16 inference.

## Key Findings

 finding_id                                                                        finding                                                                                  evidence                                                                                       why_it_matters
          1 fp16 reduced peak allocated memory by around 47% for both distilgpt2 and gpt2. distilgpt2 reduced from 343.41 MB to 181.25 MB; gpt2 reduced from 504.11 MB to 264.78 MB.                          Precision is a major memory optimization lever for LLM inference workloads.
          2                          The fp16 reduction was consistent across model sizes.                         distilgpt2 reduction was 47.22%, while gpt2 reduction was 47.48%.             This suggests that dtype-aware memory estimation can generalize across GPT-style models.
          3                  fp16 does not always give exactly 50% total memory reduction.                                     Observed reductions were around 47%, not exactly 50%. Non-parameter memory, activations, allocator behavior, and framework overhead prevent exact halving.
          4         Precision-aware estimation is required before quantization estimation.                                      fp32 and fp16 memory behavior differs significantly.                              This prepares the project for int8/int4 quantization-memory simulation.

## Main Interpretation

fp16 reduced peak allocated memory by about 47% for both distilgpt2 and gpt2 inference workloads.

The reduction is close to half but not exactly 50% because total GPU memory includes activations, framework overhead, allocator behavior, and other buffers.

Precision-aware estimation is necessary before adding int8/int4 quantization estimation.

## Next Step

Day 42 will add quantization theory and memory formulas for fp32, fp16, int8, and int4.

---

# Day 42 - Quantization Memory Theory and Formula Table

## Goal

Build the theoretical foundation for quantization-aware memory estimation.

## Context

Day 41 formalized fp32 vs fp16 precision-memory behavior.

Day 42 extends this toward fp32, fp16, int8, and int4 parameter-memory estimation.

## Files Created

- results/quantization_memory_formulas.csv
- results/quantization_parameter_memory_simulation.csv
- results/quantization_memory_reduction_summary.csv
- results/quantization_key_findings.csv
- plots/quantization_parameter_memory_by_dtype.png
- plots/quantization_memory_reduction_percent.png
- plots/quantization_distilgpt2_gpt2_comparison.png
- report/quantization_memory_theory_report.md

## Quantization Formats

dtype  bits_per_parameter  bytes_per_parameter  relative_to_fp32  theoretical_parameter_reduction_percent                                                                      notes
 fp32                  32                  4.0             1.000                                      0.0                                                   Full precision baseline.
 fp16                  16                  2.0             0.500                                     50.0         Half precision; commonly used for inference/training acceleration.
 int8                   8                  1.0             0.250                                     75.0   8-bit quantization; usually needs scales/zero-points and kernel support.
 int4                   4                  0.5             0.125                                     87.5 4-bit quantization; highly compressed but implementation overhead matters.

## Parameter Memory Simulation

model_name  num_parameters dtype  bits_per_parameter  bytes_per_parameter  estimated_parameter_memory_MB  fp32_parameter_memory_MB  parameter_memory_reduction_MB  parameter_memory_reduction_percent  theoretical_reduction_percent
distilgpt2        81912576  fp32                  32                  4.0                     312.471680                312.471680                   3.125000e-07                        1.000091e-07                            0.0
distilgpt2        81912576  fp16                  16                  2.0                     156.235840                312.471680                   1.562358e+02                        5.000000e+01                           50.0
distilgpt2        81912576  int8                   8                  1.0                      78.117920                312.471680                   2.343538e+02                        7.500000e+01                           75.0
distilgpt2        81912576  int4                   4                  0.5                      39.058960                312.471680                   2.734127e+02                        8.750000e+01                           87.5
      gpt2       124439808  fp32                  32                  4.0                     474.700195                474.700195                  -3.125000e-07                       -6.583102e-08                            0.0
      gpt2       124439808  fp16                  16                  2.0                     237.350098                474.700195                   2.373501e+02                        5.000000e+01                           50.0
      gpt2       124439808  int8                   8                  1.0                     118.675049                474.700195                   3.560251e+02                        7.500000e+01                           75.0
      gpt2       124439808  int4                   4                  0.5                      59.337524                474.700195                   4.153627e+02                        8.750000e+01                           87.5

## Reduction Summary

dtype  avg_parameter_memory_MB  avg_reduction_percent  min_reduction_percent  max_reduction_percent
 fp32               393.585938           1.708902e-08          -6.583102e-08           1.000091e-07
 fp16               196.792969           5.000000e+01           5.000000e+01           5.000000e+01
 int8                98.396484           7.500000e+01           7.500000e+01           7.500000e+01
 int4                49.198242           8.750000e+01           8.750000e+01           8.750000e+01

## Key Findings

 finding_id                                                                                 finding                                                                                                    evidence                                                                                                            why_it_matters
          1                       Parameter memory scales almost linearly with bytes per parameter.                 fp16 uses 2 bytes/parameter, int8 uses 1 byte/parameter, and int4 uses 0.5 bytes/parameter.                                             This makes quantization a direct memory optimization lever for model weights.
          2            int8 theoretically reduces parameter memory by about 75% compared with fp32.                                                    int8 uses one-fourth the storage of fp32 for parameters.                                                          This can significantly reduce inference memory for large models.
          3          int4 theoretically reduces parameter memory by about 87.5% compared with fp32.                                                    int4 uses one-eighth the storage of fp32 for parameters.                                                           This is useful for very memory-constrained deployment settings.
          4 Total runtime memory will not reduce exactly by the theoretical parameter-memory ratio.                               Day 41 showed fp16 allocated-memory reduction was about 47%, not exactly 50%. Activations, KV cache, temporary buffers, metadata, scales, zero-points, and allocator behavior also affect total memory.
          5         Quantization estimation should separate parameter memory from total GPU memory. Quantization mainly compresses weights, while activations and allocator overhead may remain fp16/fp32-like.                                                       A realistic estimator must avoid overclaiming total memory savings.

## Main Interpretation

Quantization reduces parameter memory by reducing bytes per parameter.

fp16 theoretically gives 50% parameter-memory reduction, int8 gives 75%, and int4 gives 87.5% compared with fp32.

However, total GPU memory will not reduce exactly by these percentages because runtime memory also includes activations, KV cache, temporary buffers, quantization metadata, framework overhead, and CUDA allocator behavior.

## Next Step

Day 43 will build a practical quantization-memory simulation with optional metadata overhead.

---

# Day 43 - Quantization Metadata Overhead Simulation

## Goal

Extend Day 42 quantization-memory theory by adding metadata overhead assumptions.

## Context

Day 42 created theoretical parameter-memory estimates for fp32, fp16, int8, and int4.

Day 43 makes this more realistic by adding grouped quantization metadata overhead.

## Files Created

- results/quantization_metadata_overhead_simulation.csv
- results/quantization_group_size_sensitivity.csv
- results/quantization_effective_memory_summary.csv
- results/quantization_metadata_key_findings.csv
- plots/quantization_effective_parameter_memory.png
- plots/quantization_metadata_overhead_by_group_size.png
- plots/quantization_effective_reduction_percent.png
- report/quantization_metadata_overhead_report.md

## Metadata Assumptions

        metadata_case  scale_bytes  zero_point_bytes  metadata_bytes_per_group                                               note
      scale_only_fp16          2.0               0.0                       2.0                  Stores only fp16 scale per group.
scale_plus_zero_point          2.0               2.0                       4.0 Stores fp16 scale and 2-byte zero-point per group.

## Effective Memory Summary

model_name dtype  group_size  raw_parameter_memory_MB  metadata_memory_MB  effective_parameter_memory_MB  fp32_parameter_memory_MB  effective_reduction_percent  metadata_overhead_percent_of_raw_quantized
distilgpt2  fp32         128               312.471680            0.000000                     312.471680                312.471680                      0.00000                                       0.000
distilgpt2  fp16         128               156.235840            0.000000                     156.235840                312.471680                     50.00000                                       0.000
distilgpt2  int8         128                78.117920            2.441185                      80.559105                312.471680                     74.21875                                       3.125
distilgpt2  int4         128                39.058960            2.441185                      41.500145                312.471680                     86.71875                                       6.250
      gpt2  fp32         128               474.700195            0.000000                     474.700195                474.700195                      0.00000                                       0.000
      gpt2  fp16         128               237.350098            0.000000                     237.350098                474.700195                     50.00000                                       0.000
      gpt2  int8         128               118.675049            3.708595                     122.383644                474.700195                     74.21875                                       3.125
      gpt2  int4         128                59.337524            3.708595                      63.046120                474.700195                     86.71875                                       6.250

## Group-size Sensitivity

model_name dtype  group_size  raw_parameter_memory_MB  metadata_memory_MB  effective_parameter_memory_MB  effective_reduction_percent  metadata_overhead_percent_of_raw_quantized
distilgpt2  int8          32                78.117920            9.764740                      87.882660                    71.875000                                     12.5000
distilgpt2  int8          64                78.117920            4.882370                      83.000290                    73.437500                                      6.2500
distilgpt2  int8         128                78.117920            2.441185                      80.559105                    74.218750                                      3.1250
distilgpt2  int8         256                78.117920            1.220592                      79.338512                    74.609375                                      1.5625
distilgpt2  int4          32                39.058960            9.764740                      48.823700                    84.375000                                     25.0000
distilgpt2  int4          64                39.058960            4.882370                      43.941330                    85.937500                                     12.5000
distilgpt2  int4         128                39.058960            2.441185                      41.500145                    86.718750                                      6.2500
distilgpt2  int4         256                39.058960            1.220592                      40.279552                    87.109375                                      3.1250
      gpt2  int8          32               118.675049           14.834381                     133.509430                    71.875000                                     12.5000
      gpt2  int8          64               118.675049            7.417191                     126.092239                    73.437500                                      6.2500
      gpt2  int8         128               118.675049            3.708595                     122.383644                    74.218750                                      3.1250
      gpt2  int8         256               118.675049            1.854298                     120.529346                    74.609375                                      1.5625
      gpt2  int4          32                59.337524           14.834381                      74.171906                    84.375000                                     25.0000
      gpt2  int4          64                59.337524            7.417191                      66.754715                    85.937500                                     12.5000
      gpt2  int4         128                59.337524            3.708595                      63.046120                    86.718750                                      6.2500
      gpt2  int4         256                59.337524            1.854298                      61.191822                    87.109375                                      3.1250

## Key Findings

 finding_id                                                                           finding                                                                                                      evidence                                                                                   why_it_matters
          1              Metadata overhead slightly reduces theoretical quantization savings.   With group size 128, int8 effective reduction is about 74.22% and int4 effective reduction is about 86.72%.                         Realistic quantization estimation must include scale/zero-point storage.
          2                     Metadata overhead affects int4 more than int8 proportionally. Average metadata overhead is about 3.12% of raw int8 parameter memory and 6.25% of raw int4 parameter memory. Lower-bit formats have smaller raw weight memory, so metadata becomes relatively more important.
          3                                      Larger group sizes reduce metadata overhead.                                       Group size 256 stores fewer scale/zero-point groups than group size 32.                        There is a tradeoff between quantization granularity and metadata memory.
          4        This simulation is still analytical, not real runtime quantized execution.                          The simulation adds metadata assumptions but does not load or run int8/int4 kernels.                                               It prevents overclaiming measured runtime savings.
          5 Quantization mainly reduces parameter memory, not necessarily all runtime memory.                        Activations, KV cache, temporary buffers, and allocator behavior may remain unchanged.                    A full memory estimator must separate parameter memory from total GPU memory.

## Main Interpretation

Metadata overhead slightly reduces theoretical quantization savings.

int4 is affected more than int8 proportionally because raw int4 memory is smaller.

Smaller group sizes increase metadata overhead.

This is still analytical quantization estimation, not real int8/int4 execution.

## Next Step

Day 44 will build a reusable QuantizationMemoryEstimator module.

---

# Day 44 - QuantizationMemoryEstimator Module

## Goal

Convert Day 42 and Day 43 quantization-memory formulas into a reusable estimator module.

## Files Created

- src/estimators/quantization_memory_estimator.py
- results/quantization_estimator_demo.csv
- results/quantization_estimator_summary.csv
- results/quantization_estimator_key_findings.csv
- report/quantization_memory_estimator_report.md

## Module Summary

QuantizationMemoryEstimator estimates raw parameter memory, metadata memory, effective parameter memory, and reduction versus fp32.

It supports fp32, fp16, int8, and int4.

It supports metadata cases: none, scale_only_fp16, and scale_plus_zero_point.

## Default Group-size Summary

model_name dtype  group_size  raw_parameter_memory_MB  metadata_memory_MB  effective_parameter_memory_MB  effective_reduction_percent  metadata_overhead_percent_of_raw_quantized                                                                  scope_note
distilgpt2  fp32         128               312.471680            0.000000                     312.471680                      0.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2  fp16         128               156.235840            0.000000                     156.235840                     50.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2  int8         128                78.117920            2.441185                      80.559105                     74.21875                                       3.125 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2  int4         128                39.058960            2.441185                      41.500145                     86.71875                                       6.250 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2  fp32         128               474.700195            0.000000                     474.700195                      0.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2  fp16         128               237.350098            0.000000                     237.350098                     50.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2  int8         128               118.675049            3.708595                     122.383644                     74.21875                                       3.125 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2  int4         128                59.337524            3.708595                      63.046120                     86.71875                                       6.250 Analytical parameter-memory estimate, not real quantized runtime execution.

## Key Findings

 finding_id                                                                                        finding                                                                                                 evidence                                                              why_it_matters
          1                QuantizationMemoryEstimator reproduces the Day 43 metadata-overhead simulation.             At group size 128, int8 effective memory reduction is about 74.22% and int4 is about 86.72%.            The quantization logic is now reusable instead of notebook-only.
          2 The estimator separates raw parameter memory, metadata memory, and effective parameter memory. Each output row includes raw_parameter_memory_MB, metadata_memory_MB, and effective_parameter_memory_MB.                              This avoids overclaiming quantization savings.
          3                            Metadata overhead remains proportionally larger for int4 than int8.                       At group size 128, metadata overhead averages 3.125% for int8 and 6.250% for int4.           Lower-bit quantization is more sensitive to metadata assumptions.
          4                              QuantizationMemoryEstimator is analytical, not runtime profiling.          The estimator uses parameter counts, bytes per parameter, group size, and metadata assumptions. The module should not be misrepresented as real int8/int4 CUDA measurement.

## Main Interpretation

The quantization-memory logic is now reusable project code instead of notebook-only analysis.

The estimator remains analytical and should not be described as real int8/int4 runtime profiling.

## Next Step

Day 45 will compare predicted memory under fp32, fp16, int8, and int4 using this estimator.

---

# Day 45 - Quantization Dtype Memory Comparison

## Goal

Use QuantizationMemoryEstimator to compare fp32, fp16, int8, and int4 memory estimates and connect them with measured fp32/fp16 inference results.

## Files Created

- results/quantization_dtype_comparison.csv
- results/quantization_measured_vs_analytical.csv
- results/quantization_dtype_key_findings.csv
- plots/quantization_dtype_effective_memory.png
- plots/quantization_measured_vs_parameter_memory.png
- plots/quantization_dtype_reduction_summary.png
- report/quantization_dtype_comparison_report.md

## Dtype Comparison

model_name  num_parameters dtype  bits_per_parameter  bytes_per_parameter  group_size         metadata_case  raw_parameter_memory_MB  metadata_memory_MB  effective_parameter_memory_MB  fp32_parameter_memory_MB  effective_reduction_percent  metadata_overhead_percent_of_raw_quantized                                                                  scope_note
distilgpt2        81912576  fp32                  32                  4.0         128 scale_plus_zero_point               312.471680            0.000000                     312.471680                312.471680                      0.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2        81912576  fp16                  16                  2.0         128 scale_plus_zero_point               156.235840            0.000000                     156.235840                312.471680                     50.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2        81912576  int8                   8                  1.0         128 scale_plus_zero_point                78.117920            2.441185                      80.559105                312.471680                     74.21875                                       3.125 Analytical parameter-memory estimate, not real quantized runtime execution.
distilgpt2        81912576  int4                   4                  0.5         128 scale_plus_zero_point                39.058960            2.441185                      41.500145                312.471680                     86.71875                                       6.250 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2       124439808  fp32                  32                  4.0         128 scale_plus_zero_point               474.700195            0.000000                     474.700195                474.700195                      0.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2       124439808  fp16                  16                  2.0         128 scale_plus_zero_point               237.350098            0.000000                     237.350098                474.700195                     50.00000                                       0.000 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2       124439808  int8                   8                  1.0         128 scale_plus_zero_point               118.675049            3.708595                     122.383644                474.700195                     74.21875                                       3.125 Analytical parameter-memory estimate, not real quantized runtime execution.
      gpt2       124439808  int4                   4                  0.5         128 scale_plus_zero_point                59.337524            3.708595                      63.046120                474.700195                     86.71875                                       6.250 Analytical parameter-memory estimate, not real quantized runtime execution.

## Measured vs Analytical Comparison

model_name  measured_fp32_peak_allocated_MB  measured_fp16_peak_allocated_MB  measured_allocated_reduction_percent  analytical_fp32_parameter_memory_MB  analytical_fp16_parameter_memory_MB  analytical_parameter_reduction_percent  gap_between_theoretical_and_measured_reduction_percent_points                                                                                                                                                 interpretation
distilgpt2                           343.41                           181.25                             47.220524                           312.471680                           156.235840                                    50.0                                                       2.779476 Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters.
      gpt2                           504.11                           264.78                             47.475749                           474.700195                           237.350098                                    50.0                                                       2.524251 Measured total allocated memory reduction is lower than or close to parameter-only theoretical reduction because runtime memory includes more than parameters.

## Key Findings

 finding_id                                                                                               finding                                                                                         evidence                                                                                                  why_it_matters
          1 Measured fp16 runtime memory reduction is slightly lower than theoretical parameter-memory reduction. Measured allocated reduction averaged 47.35%, while theoretical fp16 parameter reduction is 50%.             Runtime memory includes activations, KV cache, buffers, framework overhead, and allocator behavior.
          2                       The gap between theoretical and measured fp16 reduction is small but important.                                                    Average gap was about 2.65 percentage points. This shows why quantization estimators should not claim total memory savings equal to parameter-memory savings.
          3          int8 and int4 provide strong analytical parameter-memory reductions after metadata overhead.           At group size 128, int8 effective reduction averaged 74.22%, and int4 averaged 86.72%.                               Quantization is a strong candidate optimization for memory-constrained inference.
          4                                                        int8/int4 results remain analytical estimates.     The project estimates parameter and metadata memory but does not run real quantized kernels.                                           This prevents overclaiming measured runtime quantization performance.
          5                        Quantization affects parameter memory more directly than total runtime memory.                            Measured fp16 reduction was below the 50% parameter-only expectation.                                Future total-memory estimators should model activations and KV cache separately.

## Main Interpretation

Measured fp16 runtime memory reduction is slightly lower than theoretical fp16 parameter-memory reduction.

int8 and int4 provide strong analytical parameter-memory reductions after metadata overhead, but they are not measured runtime results.

Quantization mainly reduces parameter memory, while total runtime memory also includes activations, KV cache, temporary buffers, metadata, framework overhead, and allocator behavior.

## Next Step

Day 46 will create a quantization-aware total-memory approximation.

---

# Day 46 - Quantization-aware Total Memory Approximation

## Goal

Estimate total inference memory under fp16, int8, and int4 by combining measured fp32 runtime memory with analytical quantized parameter memory.

## Files Created

- results/quantization_total_memory_approximation.csv
- results/quantization_memory_decomposition.csv
- results/quantization_total_memory_key_findings.csv
- plots/quantization_total_memory_by_dtype.png
- plots/quantization_parameter_vs_non_parameter.png
- plots/quantization_total_memory_reduction.png
- report/quantization_total_memory_approximation_report.md

## Memory Decomposition

model_name  num_parameters  measured_fp32_peak_allocated_MB  fp32_parameter_memory_MB  non_parameter_memory_MB  parameter_fraction_percent  non_parameter_fraction_percent                                                                                                          interpretation
distilgpt2        81912576                           343.41                312.471680                30.938320                   90.990850                        9.009150 Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior.
      gpt2       124439808                           504.11                474.700195                29.409805                   94.165995                        5.834005 Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

## Total-memory Approximation

model_name dtype  group_size  effective_parameter_memory_MB  non_parameter_memory_assumed_MB  estimated_total_memory_MB  estimated_total_reduction_percent  measured_total_memory_MB  measured_total_reduction_percent                estimation_type                                                                           scope_note
distilgpt2  fp32         128                     312.471680                        30.938320                 343.410000                           0.000000                    343.41                          0.000000             measured_available Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2  fp16         128                     156.235840                        30.938320                 187.174160                          45.495425                    181.25                         47.220524             measured_available Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2  int8         128                      80.559105                        30.938320                 111.497425                          67.532272                       NaN                               NaN analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2  int4         128                      41.500145                        30.938320                  72.438465                          78.906128                       NaN                               NaN analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2  fp32         128                     474.700195                        29.409805                 504.110000                           0.000000                    504.11                          0.000000             measured_available Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2  fp16         128                     237.350098                        29.409805                 266.759902                          47.082997                    264.78                         47.475749             measured_available Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2  int8         128                     122.383644                        29.409805                 151.793449                          69.888824                       NaN                               NaN analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2  int4         128                      63.046120                        29.409805                  92.455924                          81.659573                       NaN                               NaN analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.

## Key Findings

 finding_id                                                                            finding                                                                                                          evidence                                                                                     why_it_matters
          1  Measured fp32 inference memory is parameter-dominated for these GPT-style models.                                                             Average non-parameter memory fraction is about 7.42%.      Quantizing parameters can significantly reduce total memory for these small GPT-style models.
          2             The total-memory approximation closely matches measured fp16 behavior.             Estimated fp16 total-memory reduction averages 46.29%, while measured fp16 reduction averages 47.35%.                           This validates the decomposition approach as a reasonable approximation.
          3 Estimated int8 total-memory reduction is lower than parameter-only int8 reduction. Estimated int8 total-memory reduction averages 68.71%, while parameter-only effective reduction was about 74.22%.                                                  Non-parameter memory limits total-memory savings.
          4 Estimated int4 total-memory reduction is lower than parameter-only int4 reduction. Estimated int4 total-memory reduction averages 80.28%, while parameter-only effective reduction was about 86.72%.  Even aggressive weight compression does not eliminate activation, KV-cache, and runtime overhead.
          5   This is still an analytical approximation, not real quantized runtime profiling.                                int8 and int4 totals are estimated using unchanged non-parameter memory from fp32. The result is useful for reasoning, but should not be overclaimed as measured quantized execution.

## Main Interpretation

Total inference memory should be separated into parameter memory and non-parameter memory.

Quantization strongly reduces parameter memory, but total memory savings are lower because non-parameter memory remains.

The approximation closely matches measured fp16 behavior, supporting the decomposition method for analytical int8/int4 reasoning.

## Next Step

Day 47 will summarize the precision and quantization adaptation phase.

---

# Day 47 - Precision and Quantization Phase Summary

## Goal

Summarize the completed precision and quantization adaptation phase.

## Files Created

- results/precision_quantization_phase_summary.csv
- results/precision_quantization_final_findings.csv
- results/precision_quantization_recommendations.csv
- report/precision_quantization_phase_summary_report.md

## Phase Summary

   day                                    phase_step                                                                                    main_output                                                                                           main_result                                                                scope
Day 41               fp32 vs fp16 precision analysis                                Measured fp16 runtime memory reduction for distilgpt2 and gpt2.                                              Average fp16 peak allocated memory reduction was 47.35%.                                 Measured fp32/fp16 inference memory.
Day 42                    Quantization memory theory                          Theoretical parameter-memory formulas for fp32, fp16, int8, and int4.                                     Theoretical parameter reductions: fp16 50%, int8 75%, int4 87.5%.                         Analytical parameter-memory estimation only.
Day 43                  Metadata overhead simulation                      Grouped quantization metadata overhead using scale + zero-point metadata.                              At group size 128, int8 effective reduction was 74.22%, int4 was 86.72%.                   Analytical parameter + metadata memory estimation.
Day 44            QuantizationMemoryEstimator module                                               Reusable estimator module under src/estimators/.    Estimator computes raw parameter memory, metadata memory, effective memory, and reduction vs fp32.                                  Reusable analytical estimator code.
Day 45             Measured vs analytical comparison         Compared measured fp32/fp16 runtime memory with analytical parameter-memory estimates.                     Measured fp16 reduction averaged 47.35%, about 2.65 points below theoretical 50%.         Measured fp32/fp16 + analytical parameter-memory comparison.
Day 46 Quantization-aware total-memory approximation Estimated total memory under fp16, int8, and int4 using parameter/non-parameter decomposition. Estimated int8 total reduction: 67.53% distilgpt2, 69.89% gpt2; int4: 78.91% distilgpt2, 81.66% gpt2. Analytical total-memory approximation, not real int8/int4 execution.

## Final Findings

 finding_id                                                                          finding                                                                                                           evidence                                                                                                           interpretation
          1    fp16 significantly reduces measured inference memory, but not by exactly 50%.                                   fp16 reduced peak allocated memory by 47.22% for distilgpt2 and 47.48% for gpt2. Total runtime memory contains parameters plus activations, KV cache, buffers, framework overhead, and allocator effects.
          2       Parameter-memory reduction is not equal to total runtime-memory reduction.                       Theoretical fp16 parameter reduction is 50%, but measured runtime reduction averaged 47.35%.                                                  Memory estimators must separate parameter and non-parameter components.
          3                           Quantization metadata reduces ideal int8/int4 savings. At group size 128, int8 effective reduction became 74.22% instead of 75%, and int4 became 86.72% instead of 87.5%.                         Scale and zero-point metadata must be included for realistic analytical quantization estimation.
          4                           int4 is more sensitive to metadata overhead than int8.                                       At group size 128, metadata overhead was 3.125% for int8 and 6.25% for int4.                             Lower-bit formats have smaller raw weight memory, so metadata becomes proportionally larger.
          5               The QuantizationMemoryEstimator makes quantization logic reusable.                                          The project now includes src/estimators/quantization_memory_estimator.py.                             Quantization estimation is now part of the project architecture, not only notebook analysis.
          6 The total-memory approximation is more realistic than parameter-only estimation.                                It decomposes measured fp32 runtime memory into parameter and non-parameter memory.                               This gives a better estimate of practical memory savings under dtype/quantization changes.
          7      int8 and int4 results are still analytical, not measured runtime profiling.                                                          No real int8/int4 quantized model was loaded or profiled.                                                              The project should not overclaim quantized runtime results.

## Recommendations

                   use_case                                                  recommendation                                                   reason                                                                  caution
Measured precision behavior                     Use fp32/fp16 measured results from Day 41.       These are actual runtime CUDA memory measurements.                               Only covers distilgpt2 and gpt2 inference.
Parameter-memory estimation Use QuantizationMemoryEstimator raw/effective parameter memory.            It includes dtype size and metadata overhead.                                  Does not represent full runtime memory.
     Total-memory reasoning       Use Day 46 quantization-aware total-memory approximation.         It separates parameter and non-parameter memory.                          Assumes non-parameter memory remains unchanged.
           int8/int4 claims                             Label as analytical estimates only.      No real int8/int4 quantized execution was measured.                Do not claim runtime memory, latency, or quality results.
       Next technical phase                             Move to sparsity memory estimation. Precision and quantization phase is now complete enough. Do not keep over-tweaking quantization without new measured experiments.

## Main Interpretation

The precision and quantization phase shows that dtype and weight compression can significantly reduce memory.

However, parameter-memory reduction does not directly equal total runtime-memory reduction.

Measured fp16 runtime reduction averaged 47.35%, while theoretical fp16 parameter-memory reduction is 50%.

int8/int4 estimates remain analytical, not measured runtime profiling.

## Next Step

Day 48 will start sparsity memory estimation.

---

# Day 48 - Sparsity Memory Theory

## Goal

Start the sparsity memory estimation phase by analytically estimating sparse parameter memory.

## Files Created

- results/sparsity_memory_formulas.csv
- results/sparsity_parameter_memory_simulation.csv
- results/sparsity_memory_reduction_summary.csv
- results/sparsity_key_findings.csv
- plots/sparsity_parameter_memory_by_level.png
- plots/sparsity_memory_reduction_percent.png
- plots/sparsity_dense_vs_sparse_comparison_distilgpt2.png
- plots/sparsity_dense_vs_sparse_comparison_gpt2.png
- report/sparsity_memory_theory_report.md

## Sparsity Simulation

model_name  num_parameters  sparsity_percent  nonzero_fraction  nonzero_parameters  dense_fp32_parameter_memory_MB  sparse_value_memory_MB  sparse_index_memory_MB  sparse_total_parameter_memory_MB  sparse_reduction_MB  sparse_reduction_percent                                                                      scope_note
distilgpt2        81912576               0.0              1.00            81912576                      312.471680              312.471680              312.471680                        624.943359          -312.471680                    -100.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576              25.0              0.75            61434432                      312.471680              234.353760              234.353760                        468.707520          -156.235840                     -50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576              50.0              0.50            40956288                      312.471680              156.235840              156.235840                        312.471680             0.000000                       0.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576              75.0              0.25            20478144                      312.471680               78.117920               78.117920                        156.235840           156.235840                      50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576              90.0              0.10             8191257                      312.471680               31.247168               31.247168                         62.494336           249.977344                      80.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808               0.0              1.00           124439808                      474.700195              474.700195              474.700195                        949.400391          -474.700195                    -100.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808              25.0              0.75            93329856                      474.700195              356.025146              356.025146                        712.050293          -237.350098                     -50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808              50.0              0.50            62219904                      474.700195              237.350098              237.350098                        474.700195             0.000000                       0.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808              75.0              0.25            31109952                      474.700195              118.675049              118.675049                        237.350098           237.350098                      50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808              90.0              0.10            12443980                      474.700195               47.470020               47.470020                         94.940039           379.760156                      80.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.

## Reduction Summary

 sparsity_percent  avg_sparse_total_parameter_memory_MB  avg_reduction_percent  min_reduction_percent  max_reduction_percent
              0.0                            787.171875                 -100.0                 -100.0                 -100.0
             25.0                            590.378906                  -50.0                  -50.0                  -50.0
             50.0                            393.585938                    0.0                    0.0                    0.0
             75.0                            196.792969                   50.0                   50.0                   50.0
             90.0                             78.717187                   80.0                   80.0                   80.0

## Key Findings

 finding_id                                                                        finding                                                                                                evidence                                                                                               why_it_matters
          1                                    Unstructured sparsity needs index metadata.                                      The sparse estimate stores both nonzero values and index metadata.                                Sparse memory savings are lower than the raw zero-weight percentage suggests.
          2                           Low sparsity can use more memory than dense storage. With value_bytes=4 and index_bytes=4, sparse storage at 25% sparsity is larger than dense fp32 storage. Sparsity is not automatically a memory win unless sparsity is high enough or the sparse format is efficient.
          3 Around 50% sparsity is the break-even point in this simple unstructured model.                           At 50% sparsity, nonzero values plus indices roughly equal dense fp32 memory.                            Below this level, sparse storage overhead can cancel the benefit of zero weights.
          4                       High sparsity can reduce parameter memory significantly.             At 75% and 90% sparsity, sparse parameter memory becomes lower than dense parameter memory.           Sparsity can be useful for memory reduction when sparsity is high and sparse storage is supported.
          5      This is analytical storage estimation, not real sparse runtime profiling.                                     The simulation does not run sparse CUDA kernels or measure latency.                      Sparse runtime benefits depend on hardware, kernels, sparse format, and workload shape.

## Main Interpretation

Sparsity does not automatically reduce memory because sparse storage needs index metadata.

At low sparsity, sparse storage can be worse than dense storage.

In this simple unstructured model, 50% sparsity is roughly the break-even point.

High sparsity levels such as 75% and 90% can reduce parameter memory significantly.

## Next Step

Day 49 will build a reusable SparsityMemoryEstimator module.

---

# Day 49 - SparsityMemoryEstimator Module

## Goal

Convert Day 48 sparsity-memory formulas into a reusable estimator module.

## Files Created

- src/estimators/sparsity_memory_estimator.py
- results/sparsity_estimator_demo.csv
- results/sparsity_estimator_summary.csv
- results/sparsity_estimator_key_findings.csv
- report/sparsity_memory_estimator_report.md

## Estimator Demo

model_name  num_parameters             storage_type  sparsity_percent  nonzero_fraction  nonzero_parameters  value_bytes  index_bytes_per_nonzero  uses_index_metadata  dense_fp32_parameter_memory_MB  sparse_value_memory_MB  sparse_index_memory_MB  sparse_total_parameter_memory_MB  sparse_reduction_percent  sparse_overhead_vs_dense_percent                                                                      scope_note
distilgpt2        81912576 unstructured_sparse_fp32               0.0              1.00            81912576          4.0                      4.0                 True                      312.471680              312.471680              312.471680                        624.943359                    -100.0                             100.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576 unstructured_sparse_fp32              25.0              0.75            61434432          4.0                      4.0                 True                      312.471680              234.353760              234.353760                        468.707520                     -50.0                              50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576 unstructured_sparse_fp32              50.0              0.50            40956288          4.0                      4.0                 True                      312.471680              156.235840              156.235840                        312.471680                       0.0                               0.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576 unstructured_sparse_fp32              75.0              0.25            20478144          4.0                      4.0                 True                      312.471680               78.117920               78.117920                        156.235840                      50.0                             -50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
distilgpt2        81912576 unstructured_sparse_fp32              90.0              0.10             8191257          4.0                      4.0                 True                      312.471680               31.247168               31.247168                         62.494336                      80.0                             -80.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808 unstructured_sparse_fp32               0.0              1.00           124439808          4.0                      4.0                 True                      474.700195              474.700195              474.700195                        949.400391                    -100.0                             100.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808 unstructured_sparse_fp32              25.0              0.75            93329856          4.0                      4.0                 True                      474.700195              356.025146              356.025146                        712.050293                     -50.0                              50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808 unstructured_sparse_fp32              50.0              0.50            62219904          4.0                      4.0                 True                      474.700195              237.350098              237.350098                        474.700195                       0.0                               0.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808 unstructured_sparse_fp32              75.0              0.25            31109952          4.0                      4.0                 True                      474.700195              118.675049              118.675049                        237.350098                      50.0                             -50.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.
      gpt2       124439808 unstructured_sparse_fp32              90.0              0.10            12443980          4.0                      4.0                 True                      474.700195               47.470020               47.470020                         94.940039                      80.0                             -80.0 Analytical sparse parameter-memory estimate, not real sparse runtime execution.

## Summary

 sparsity_percent  avg_sparse_total_parameter_memory_MB  avg_sparse_reduction_percent  min_sparse_reduction_percent  max_sparse_reduction_percent
              0.0                            787.171875                        -100.0                        -100.0                        -100.0
             25.0                            590.378906                         -50.0                         -50.0                         -50.0
             50.0                            393.585938                           0.0                           0.0                           0.0
             75.0                            196.792969                          50.0                          50.0                          50.0
             90.0                             78.717187                          80.0                          80.0                          80.0

## Key Findings

 finding_id                                                                    finding                                                                                          evidence                                                                 why_it_matters
          1 SparsityMemoryEstimator reproduces the Day 48 analytical sparsity results. 50% sparsity breaks even, 75% sparsity gives 50% reduction, and 90% sparsity gives 80% reduction. Sparsity logic is now reusable project code instead of notebook-only analysis.
          2            The estimator separates value memory and index metadata memory.                       Each output row includes sparse_value_memory_MB and sparse_index_memory_MB.                              This prevents overclaiming sparse memory savings.
          3                            Unstructured sparsity has a high metadata cost.    At 0% sparsity, sparse storage uses 2x dense memory because it stores both values and indices.                 Sparse storage is not automatically better than dense storage.
          4              SparsityMemoryEstimator is analytical, not runtime profiling.                 The estimator uses parameter count, sparsity level, value bytes, and index bytes.           The module should not be described as measured sparse CUDA behavior.

## Main Interpretation

The sparsity-memory logic is now reusable project code instead of notebook-only analysis.

The estimator remains analytical and should not be described as real sparse runtime profiling.

## Next Step

Day 50 will create a sparsity-aware total-memory approximation.

---

# Day 50 - Sparsity-aware Total Memory Approximation

## Goal

Estimate total inference memory under different sparsity levels by combining measured fp32 runtime memory with analytical sparse parameter memory.

## Files Created

- results/sparsity_total_memory_approximation.csv
- results/sparsity_memory_decomposition.csv
- results/sparsity_total_memory_key_findings.csv
- plots/sparsity_total_memory_by_level.png
- plots/sparsity_parameter_vs_non_parameter.png
- plots/sparsity_total_memory_reduction.png
- report/sparsity_total_memory_approximation_report.md

## Memory Decomposition

model_name  num_parameters  measured_fp32_peak_allocated_MB  dense_fp32_parameter_memory_MB  non_parameter_memory_MB  parameter_fraction_percent  non_parameter_fraction_percent                                                                                                          interpretation
distilgpt2        81912576                           343.41                      312.471680                30.938320                   90.990850                        9.009150 Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior.
      gpt2       124439808                           504.11                      474.700195                29.409805                   94.165995                        5.834005 Non-parameter memory approximates activations, KV cache, temporary buffers, framework overhead, and allocator behavior.

## Total-memory Approximation

model_name  sparsity_percent  nonzero_fraction  nonzero_parameters  dense_fp32_parameter_memory_MB  sparse_value_memory_MB  sparse_index_memory_MB  sparse_total_parameter_memory_MB  non_parameter_memory_assumed_MB  estimated_total_memory_MB  measured_fp32_total_memory_MB  estimated_total_reduction_percent                estimation_type                                                                           scope_note
distilgpt2               0.0              1.00            81912576                      312.471680              312.471680              312.471680                        624.943359                        30.938320                 655.881680                         343.41                         -90.990850 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2              25.0              0.75            61434432                      312.471680              234.353760              234.353760                        468.707520                        30.938320                 499.645840                         343.41                         -45.495425 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2              50.0              0.50            40956288                      312.471680              156.235840              156.235840                        312.471680                        30.938320                 343.410000                         343.41                           0.000000 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2              75.0              0.25            20478144                      312.471680               78.117920               78.117920                        156.235840                        30.938320                 187.174160                         343.41                          45.495425 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
distilgpt2              90.0              0.10             8191257                      312.471680               31.247168               31.247168                         62.494336                        30.938320                  93.432656                         343.41                          72.792680 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2               0.0              1.00           124439808                      474.700195              474.700195              474.700195                        949.400391                        29.409805                 978.810195                         504.11                         -94.165995 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2              25.0              0.75            93329856                      474.700195              356.025146              356.025146                        712.050293                        29.409805                 741.460098                         504.11                         -47.082997 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2              50.0              0.50            62219904                      474.700195              237.350098              237.350098                        474.700195                        29.409805                 504.110000                         504.11                           0.000000 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2              75.0              0.25            31109952                      474.700195              118.675049              118.675049                        237.350098                        29.409805                 266.759902                         504.11                          47.082997 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.
      gpt2              90.0              0.10            12443980                      474.700195               47.470020               47.470020                         94.940039                        29.409805                 124.349844                         504.11                          75.332796 analytical_total_approximation Total-memory approximation assumes non-parameter memory remains unchanged from fp32.

## Key Findings

 finding_id                                                                                finding                                                                                                      evidence                                                                                             why_it_matters
          1 Measured fp32 inference memory is parameter-dominated for the tested GPT-style models.                                                         Average non-parameter memory fraction is about 7.42%. Sparse parameter compression can significantly affect estimated total memory when sparsity is high enough.
          2              Low sparsity can increase total memory under unstructured sparse storage.                                         At 25% sparsity, average estimated total-memory reduction is -46.29%.                                  Index metadata can outweigh savings when too many weights remain nonzero.
          3         50% sparsity is approximately the total-memory break-even point in this model.                                           At 50% sparsity, average estimated total-memory reduction is 0.00%.                                      This matches the parameter-memory break-even behavior from Day 48/49.
          4                                   High sparsity gives meaningful total-memory savings.                              At 75% sparsity, average total-memory reduction is 46.29%; at 90%, it is 74.06%.                 Sparsity only becomes useful when sparsity is high enough and sparse storage is supported.
          5          This is still an analytical approximation, not real sparse runtime profiling. Sparse total memory is estimated using unchanged non-parameter memory and analytical sparse parameter memory.        Real sparse runtime depends on sparse kernels, hardware support, sparse format, and workload shape.

## Main Interpretation

Total memory should be separated into parameter and non-parameter memory.

Sparsity reduces parameter memory only when sparsity is high enough to overcome index metadata overhead.

Low sparsity can increase total memory under unstructured sparse storage.

At around 50% sparsity, the current unstructured sparse model breaks even.

## Next Step

Day 51 will summarize the sparsity phase and compare sparsity with quantization.

---

# Day 51 - Sparsity Phase Summary and Quantization Comparison

## Goal

Summarize the sparsity memory estimation phase and compare sparsity with quantization.

## Files Created

- results/sparsity_phase_summary.csv
- results/sparsity_vs_quantization_comparison.csv
- results/sparsity_final_findings.csv
- results/sparsity_recommendations.csv
- report/sparsity_phase_summary_report.md

## Sparsity Phase Summary

   day                                phase_step                                                                    main_output                                                                                                      main_result                                                                     scope
Day 48                    Sparsity memory theory                      Analytical unstructured sparse parameter-memory formulas.                     50% sparsity breaks even; 75% gives 50% parameter-memory reduction; 90% gives 80% reduction.                              Analytical parameter-memory estimation only.
Day 49            SparsityMemoryEstimator module          Reusable estimator under src/estimators/sparsity_memory_estimator.py. Module estimates dense memory, sparse value memory, index metadata, total sparse memory, and reduction vs dense.                                       Reusable analytical estimator code.
Day 50 Sparsity-aware total-memory approximation Combined measured fp32 runtime memory with analytical sparse parameter memory.                50% sparsity breaks even in total memory; 75% and 90% sparsity reduce total memory significantly. Analytical total-memory approximation, not real sparse runtime execution.

## Sparsity vs Quantization Comparison

                   method                      compression_type runtime_measured                                             analytical_or_measured  avg_total_memory_reduction_percent  distilgpt2_reduction_percent  gpt2_reduction_percent                                                                                     main_tradeoff
           fp16 precision                 lower precision dtype  partly measured measured fp32/fp16 runtime memory + analytical total approximation                           46.289211                     45.495425               47.082997            Usually practical and widely supported, but savings are lower than parameter-only 50%.
        int8 quantization   lower precision weight quantization               no                                           analytical estimate only                           68.710548                     67.532272               69.888824               Strong memory reduction, but requires real quantized execution support to validate.
        int4 quantization   lower precision weight quantization               no                                           analytical estimate only                           80.282851                     78.906128               81.659573    Largest estimated memory reduction, but metadata, packing, quality, and kernel support matter.
75% unstructured sparsity remove zero weights but store indices               no                                           analytical estimate only                           46.289211                     45.495425               47.082997                            Only useful after enough sparsity to overcome index metadata overhead.
90% unstructured sparsity remove zero weights but store indices               no                                           analytical estimate only                           74.062738                     72.792680               75.332796 Strong estimated reduction, but real sparse speed/memory depends heavily on hardware and kernels.

## Final Findings

 finding_id                                                                             finding                                                                                                              evidence                                                                                   interpretation
          1                        Unstructured sparsity is not automatically memory-efficient. At 0% and 25% sparsity, estimated total memory is worse than dense fp32 because each nonzero stores value plus index.                   Sparsity needs high zero density or efficient sparse formats to become useful.
          2 50% sparsity is the break-even point in the current unstructured sparse fp32 model.                                 At 50% sparsity, estimated total-memory reduction is 0% for both distilgpt2 and gpt2.                                         The index metadata cancels out the value-memory savings.
          3      75% sparsity gives similar total-memory savings to fp16 in this approximation.                                     Average 75% sparsity reduction is 46.29%, while average fp16 reduction is 46.29%.                 Sparse storage needs quite high sparsity to compete with simple dtype reduction.
          4      90% sparsity gives strong memory reduction but still needs runtime validation.                                                                Average 90% sparsity total-memory reduction is 74.06%. The analytical estimate is promising, but real sparse execution depends on kernels and hardware.
          5       Quantization is easier to reason about for memory than unstructured sparsity.                           Quantization reduces bytes per value, while unstructured sparsity also adds index metadata.                            Sparsity has a more complicated memory tradeoff than dtype reduction.
          6                   The sparsity phase is analytical, not measured runtime profiling.                   No sparse CUDA kernels, sparse matrix multiplications, or compressed model execution were measured.                       The project should clearly label sparsity results as analytical estimates.

## Recommendations

                      use_case                                                 recommendation                                                                             reason                                                               caution
            Safe project claim Describe sparsity results as analytical memory estimates only.                                     No real sparse runtime execution was measured.           Do not claim sparse speedup or measured sparse CUDA memory.
Memory optimization comparison       Present fp16 as the strongest measured precision result.             fp16 has actual measured runtime memory data from earlier experiments.                      Only validated on distilgpt2 and gpt2 inference.
       Sparsity interpretation           Emphasize metadata overhead and break-even sparsity. The main systems insight is that sparsity can be worse than dense at low sparsity.                          Avoid saying sparsity always reduces memory.
                   Future work           Mention real sparse kernel profiling as future work.                 Runtime behavior depends on sparse formats, hardware, and kernels.           Analytical estimates cannot prove real deployment behavior.
            Next project phase         Move to model-parallel memory partitioning simulation.   Precision, quantization, and sparsity adaptation phases are now complete enough. Do not overextend sparsity without real sparse execution experiments.

## Main Interpretation

The sparsity phase shows that unstructured sparsity is not automatically memory-efficient.

Index metadata can make sparse storage worse than dense storage at low sparsity.

At 50% sparsity, the current unstructured sparse fp32 format breaks even.

At 75% and 90% sparsity, sparse storage gives useful estimated total-memory savings.

These results remain analytical and should not be described as measured sparse runtime profiling.

## Next Step

Day 52 will start model-parallel memory partitioning simulation.

---

# Day 52 - Model-parallel Memory Theory

## Goal

Start the model-parallel memory estimation phase by analytically estimating per-device parameter memory under model partitioning.

## Files Created

- results/model_parallel_memory_formulas.csv
- results/model_parallel_parameter_partitioning.csv
- results/model_parallel_key_findings.csv
- plots/model_parallel_parameter_memory_by_devices_fp32.png
- plots/model_parallel_parameter_memory_by_devices_fp16.png
- plots/model_parallel_memory_reduction_percent_fp32.png
- plots/model_parallel_memory_reduction_percent_fp16.png
- plots/model_parallel_replication_overhead.png
- report/model_parallel_memory_theory_report.md

## Simulation

model_name dtype  num_devices  total_parameter_memory_MB  ideal_partitioned_parameter_memory_MB  replicated_overhead_memory_MB  communication_buffer_memory_MB  total_overhead_memory_MB  estimated_per_device_memory_MB  ideal_reduction_percent  effective_reduction_percent  overhead_fraction_of_total_percent                                                                 scope_note
distilgpt2  fp32            1                 312.471680                             312.471680                      15.623584                        9.374150                 24.997734                      337.469414                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp32            2                 312.471680                             156.235840                      15.623584                        9.374150                 24.997734                      181.233574                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp32            4                 312.471680                              78.117920                      15.623584                        9.374150                 24.997734                      103.115654                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp32            8                 312.471680                              39.058960                      15.623584                        9.374150                 24.997734                       64.056694                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp16            1                 156.235840                             156.235840                       7.811792                        4.687075                 12.498867                      168.734707                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp16            2                 156.235840                              78.117920                       7.811792                        4.687075                 12.498867                       90.616787                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp16            4                 156.235840                              39.058960                       7.811792                        4.687075                 12.498867                       51.557827                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
distilgpt2  fp16            8                 156.235840                              19.529480                       7.811792                        4.687075                 12.498867                       32.028347                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp32            1                 474.700195                             474.700195                      23.735010                       14.241006                 37.976016                      512.676211                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp32            2                 474.700195                             237.350098                      23.735010                       14.241006                 37.976016                      275.326113                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp32            4                 474.700195                             118.675049                      23.735010                       14.241006                 37.976016                      156.651064                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp32            8                 474.700195                              59.337524                      23.735010                       14.241006                 37.976016                       97.313540                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp16            1                 237.350098                             237.350098                      11.867505                        7.120503                 18.988008                      256.338106                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp16            2                 237.350098                             118.675049                      11.867505                        7.120503                 18.988008                      137.663057                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp16            4                 237.350098                              59.337525                      11.867505                        7.120503                 18.988008                       78.325532                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.
      gpt2  fp16            8                 237.350098                              29.668762                      11.867505                        7.120503                 18.988008                       48.656770                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real distributed execution.

## Key Findings

 finding_id                                                                                             finding                                                                                            evidence                                                                                    why_it_matters
          1 Model parallelism reduces per-device parameter memory by partitioning model weights across devices.                            Ideal partitioned memory scales as total_parameter_memory / num_devices.                      This allows larger models to fit when one device cannot hold the full model.
          2                         Effective memory reduction is lower than ideal scaling because of overhead.                       The simulation includes 8.0% combined replication and communication overhead.                                                   Real model parallelism is not perfectly linear.
          3                                  Two-device partitioning gives useful but limited memory reduction.                                 Average fp32 effective per-device reduction at 2 devices is 42.00%.                           Two devices help, but overhead still consumes a visible part of memory.
          4                                Four and eight devices provide stronger per-device memory reduction.                    Average fp32 effective reduction is 67.00% at 4 devices and 79.50% at 8 devices.    More devices reduce per-device weight storage, but communication and replication costs remain.
          5                         This is analytical model-parallel planning, not real distributed profiling. No tensor parallelism, pipeline parallelism, NCCL communication, or multi-GPU runtime was executed. The results should be presented as memory-estimation logic, not measured distributed performance.

## Main Interpretation

Model parallelism reduces per-device parameter memory by splitting model parameters across devices.

Ideal memory reduction scales with the number of devices, but real systems have replicated overhead and communication buffers.

This simulation is analytical and should not be described as real multi-GPU profiling.

## Next Step

Day 53 will build a reusable ModelParallelMemoryEstimator module.

---

# Day 53 - ModelParallelMemoryEstimator Module

## Goal

Convert Day 52 model-parallel memory formulas into a reusable estimator module.

## Files Created

- src/estimators/model_parallel_memory_estimator.py
- results/model_parallel_estimator_demo.csv
- results/model_parallel_estimator_summary.csv
- results/model_parallel_estimator_key_findings.csv
- report/model_parallel_memory_estimator_report.md

## Estimator Demo

model_name  num_parameters dtype  num_devices  total_parameter_memory_MB  ideal_partitioned_parameter_memory_MB  replication_overhead_percent  communication_buffer_percent  replicated_overhead_memory_MB  communication_buffer_memory_MB  total_overhead_memory_MB  estimated_per_device_memory_MB  ideal_reduction_percent  effective_reduction_percent  overhead_fraction_of_total_percent                                                               scope_note
distilgpt2        81912576  fp32            1                 312.471680                             312.471680                           5.0                           3.0                      15.623584                        9.374150                 24.997734                      337.469414                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp32            2                 312.471680                             156.235840                           5.0                           3.0                      15.623584                        9.374150                 24.997734                      181.233574                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp32            4                 312.471680                              78.117920                           5.0                           3.0                      15.623584                        9.374150                 24.997734                      103.115654                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp32            8                 312.471680                              39.058960                           5.0                           3.0                      15.623584                        9.374150                 24.997734                       64.056694                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp16            1                 156.235840                             156.235840                           5.0                           3.0                       7.811792                        4.687075                 12.498867                      168.734707                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp16            2                 156.235840                              78.117920                           5.0                           3.0                       7.811792                        4.687075                 12.498867                       90.616787                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp16            4                 156.235840                              39.058960                           5.0                           3.0                       7.811792                        4.687075                 12.498867                       51.557827                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
distilgpt2        81912576  fp16            8                 156.235840                              19.529480                           5.0                           3.0                       7.811792                        4.687075                 12.498867                       32.028347                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp32            1                 474.700195                             474.700195                           5.0                           3.0                      23.735010                       14.241006                 37.976016                      512.676211                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp32            2                 474.700195                             237.350098                           5.0                           3.0                      23.735010                       14.241006                 37.976016                      275.326113                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp32            4                 474.700195                             118.675049                           5.0                           3.0                      23.735010                       14.241006                 37.976016                      156.651064                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp32            8                 474.700195                              59.337524                           5.0                           3.0                      23.735010                       14.241006                 37.976016                       97.313540                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp16            1                 237.350098                             237.350098                           5.0                           3.0                      11.867505                        7.120503                 18.988008                      256.338105                      0.0                         -8.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp16            2                 237.350098                             118.675049                           5.0                           3.0                      11.867505                        7.120503                 18.988008                      137.663057                     50.0                         42.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp16            4                 237.350098                              59.337524                           5.0                           3.0                      11.867505                        7.120503                 18.988008                       78.325532                     75.0                         67.0                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.
      gpt2       124439808  fp16            8                 237.350098                              29.668762                           5.0                           3.0                      11.867505                        7.120503                 18.988008                       48.656770                     87.5                         79.5                                 8.0 Analytical model-parallel memory estimate, not real multi-GPU execution.

## Summary

dtype  num_devices  avg_estimated_per_device_memory_MB  avg_ideal_reduction_percent  avg_effective_reduction_percent  avg_overhead_fraction_percent
 fp16            1                          212.536406                          0.0                             -8.0                            8.0
 fp16            2                          114.139922                         50.0                             42.0                            8.0
 fp16            4                           64.941680                         75.0                             67.0                            8.0
 fp16            8                           40.342559                         87.5                             79.5                            8.0
 fp32            1                          425.072812                          0.0                             -8.0                            8.0
 fp32            2                          228.279844                         50.0                             42.0                            8.0
 fp32            4                          129.883359                         75.0                             67.0                            8.0
 fp32            8                           80.685117                         87.5                             79.5                            8.0

## Key Findings

 finding_id                                                                          finding                                                                                                                    evidence                                                                      why_it_matters
          1 ModelParallelMemoryEstimator reproduces Day 52 analytical partitioning behavior.                            Effective fp32 reductions are 42.00% at 2 devices, 67.00% at 4 devices, and 79.50% at 8 devices.                           Model-parallel memory logic is now reusable project code.
          2           The estimator separates ideal partitioned memory from overhead memory. Each row includes ideal_partitioned_parameter_memory_MB, replicated_overhead_memory_MB, and communication_buffer_memory_MB.                  This avoids pretending model-parallel scaling is perfectly linear.
          3                  Combined overhead reduces ideal scaling by 8 percentage points.                                            The estimator uses 5% replication overhead and 3% communication buffer overhead. Distributed memory planning must include communication and replicated-memory costs.
          4            The module supports fp32, fp16, and bf16 parameter-memory estimation.                                                                            supported_dtypes() returns fp32, fp16, and bf16.           The estimator can be combined later with precision-aware memory planning.
          5      This is analytical model-parallel estimation, not real multi-GPU profiling.                       No tensor parallelism, pipeline parallelism, NCCL communication, or distributed runtime was executed.             The project should claim model-parallel simulation, not implementation.

## Main Interpretation

The model-parallel memory logic is now reusable project code instead of notebook-only analysis.

The estimator separates ideal partitioned memory from replicated and communication overhead.

The estimator remains analytical and should not be described as real multi-GPU profiling.

## Next Step

Day 54 will build a model-parallel total-memory approximation and compare it with quantization and sparsity.

---

# Day 54 - Model-parallel Total-memory Approximation

## Goal

Estimate total per-device memory under model parallelism and compare model parallelism with quantization and sparsity.

## Files Created

- results/model_parallel_total_memory_approximation.csv
- results/model_parallel_memory_decomposition.csv
- results/model_parallel_vs_other_optimizations.csv
- results/model_parallel_total_memory_key_findings.csv
- plots/model_parallel_total_memory_by_devices_fp32.png
- plots/model_parallel_total_memory_by_devices_fp16.png
- plots/model_parallel_total_memory_reduction_fp32.png
- plots/model_parallel_total_memory_reduction_fp16.png
- plots/model_parallel_vs_optimization_reduction.png
- report/model_parallel_total_memory_approximation_report.md

## Memory Decomposition

model_name  num_parameters  measured_fp32_peak_allocated_MB  dense_fp32_parameter_memory_MB  non_parameter_memory_MB  parameter_fraction_percent  non_parameter_fraction_percent                                                               assumption
distilgpt2        81912576                           343.41                      312.471680                30.938320                   90.990850                        9.009150 Non-parameter memory is assumed replicated and unchanged across devices.
      gpt2       124439808                           504.11                      474.700195                29.409805                   94.165995                        5.834005 Non-parameter memory is assumed replicated and unchanged across devices.

## Model-parallel Total-memory Approximation

model_name dtype  num_devices  measured_fp32_total_memory_MB  non_parameter_memory_assumed_MB  total_parameter_memory_MB  ideal_partitioned_parameter_memory_MB  replicated_overhead_memory_MB  communication_buffer_memory_MB  model_parallel_per_device_parameter_memory_MB  estimated_total_per_device_memory_MB  ideal_parameter_reduction_percent  effective_parameter_reduction_percent  estimated_total_reduction_vs_fp32_percent                       estimation_type                                                                              scope_note
distilgpt2  fp32            1                         343.41                        30.938320                 312.471680                             312.471680                      15.623584                        9.374150                                     337.469414                            368.407734                                0.0                                   -8.0                                  -7.279268 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp32            2                         343.41                        30.938320                 312.471680                             156.235840                      15.623584                        9.374150                                     181.233574                            212.171895                               50.0                                   42.0                                  38.216157 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp32            4                         343.41                        30.938320                 312.471680                              78.117920                      15.623584                        9.374150                                     103.115654                            134.053975                               75.0                                   67.0                                  60.963870 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp32            8                         343.41                        30.938320                 312.471680                              39.058960                      15.623584                        9.374150                                      64.056694                             94.995015                               87.5                                   79.5                                  72.337726 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp16            1                         343.41                        30.938320                 156.235840                             156.235840                       7.811792                        4.687075                                     168.734707                            199.673027                                0.0                                   -8.0                                  41.855791 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp16            2                         343.41                        30.938320                 156.235840                              78.117920                       7.811792                        4.687075                                      90.616787                            121.555107                               50.0                                   42.0                                  64.603504 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp16            4                         343.41                        30.938320                 156.235840                              39.058960                       7.811792                        4.687075                                      51.557827                             82.496147                               75.0                                   67.0                                  75.977360 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
distilgpt2  fp16            8                         343.41                        30.938320                 156.235840                              19.529480                       7.811792                        4.687075                                      32.028347                             62.966667                               87.5                                   79.5                                  81.664288 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp32            1                         504.11                        29.409805                 474.700195                             474.700195                      23.735010                       14.241006                                     512.676211                            542.086016                                0.0                                   -8.0                                  -7.533280 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp32            2                         504.11                        29.409805                 474.700195                             237.350098                      23.735010                       14.241006                                     275.326113                            304.735918                               50.0                                   42.0                                  39.549718 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp32            4                         504.11                        29.409805                 474.700195                             118.675049                      23.735010                       14.241006                                     156.651064                            186.060869                               75.0                                   67.0                                  63.091216 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp32            8                         504.11                        29.409805                 474.700195                              59.337524                      23.735010                       14.241006                                      97.313540                            126.723345                               87.5                                   79.5                                  74.861966 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp16            1                         504.11                        29.409805                 237.350098                             237.350098                      11.867505                        7.120503                                     256.338105                            285.747910                                0.0                                   -8.0                                  43.316358 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp16            2                         504.11                        29.409805                 237.350098                             118.675049                      11.867505                        7.120503                                     137.663057                            167.072861                               50.0                                   42.0                                  66.857856 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp16            4                         504.11                        29.409805                 237.350098                              59.337524                      11.867505                        7.120503                                      78.325532                            107.735337                               75.0                                   67.0                                  78.628605 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.
      gpt2  fp16            8                         504.11                        29.409805                 237.350098                              29.668762                      11.867505                        7.120503                                      48.656770                             78.066575                               87.5                                   79.5                                  84.513980 analytical_total_memory_approximation Assumes non-parameter memory is replicated and unchanged; not real multi-GPU profiling.

## Comparison with Other Optimizations

                            method             optimization_type runtime_measured  avg_total_reduction_percent  distilgpt2_reduction_percent  gpt2_reduction_percent                                                                  main_tradeoff
                    fp16 precision           precision reduction  partly measured                    46.289211                     45.495425               47.082997                 Practical and widely supported; measured fp16 behavior exists.
                 int8 quantization           weight quantization               no                    68.710548                     67.532272               69.888824 Strong analytical memory reduction; needs real quantized execution validation.
                 int4 quantization           weight quantization               no                    80.282851                     78.906128               81.659573     Largest analytical quantization saving; quality and kernel support matter.
         75% unstructured sparsity                sparse storage               no                    46.289211                     45.495425               47.082997                  Useful only after sparsity overcomes index metadata overhead.
         90% unstructured sparsity                sparse storage               no                    74.062738                     72.792680               75.332796     Strong analytical saving; real sparse runtime depends on kernels/hardware.
fp32 model parallelism - 2 devices             model parallelism               no                    38.882937                     38.216157               39.549718        Reduces per-device memory but needs multiple devices and communication.
fp32 model parallelism - 4 devices             model parallelism               no                    62.027543                     60.963870               63.091216     Stronger per-device memory reduction; more communication and coordination.
fp32 model parallelism - 8 devices             model parallelism               no                    73.599846                     72.337726               74.861966           Large per-device reduction but higher distributed-system complexity.
fp16 model parallelism - 4 devices precision + model parallelism               no                    77.302983                     75.977360               78.628605              Combines dtype reduction and model partitioning; analytical only.

## Key Findings

 finding_id                                                                                                    finding                                                                                                  evidence                                                                                 why_it_matters
          1                        Model parallelism reduces per-device total memory by partitioning parameter memory. Average fp32 total-memory reduction is 38.88% at 2 devices, 62.03% at 4 devices, and 73.60% at 8 devices.              This helps estimate whether a model can fit on smaller per-device memory budgets.
          2                              Total-memory reduction is lower than parameter-only model-parallel reduction.                                  Non-parameter memory is assumed replicated and unchanged across devices.              Activation, KV-cache, framework, and allocator memory limit total-memory scaling.
          3                       Combining fp16 with model parallelism gives stronger estimated per-device reduction.                                               Average fp16 + 4-device model-parallel reduction is 77.30%.                    Multiple optimization techniques can be combined for larger memory savings.
          4 Quantization can be more memory-efficient than low-device-count model parallelism in analytical estimates.                     Average int8 reduction is 68.71%, while fp32 2-device model parallelism gives 38.88%.                                 Different optimization strategies solve different constraints.
          5                                  Model parallelism has a different tradeoff than quantization or sparsity.              Model parallelism reduces per-device memory but requires multiple devices and communication. It is useful when a model is too large for one device, but adds distributed-system complexity.
          6                                                   This comparison is analytical, not runtime benchmarking.              Quantization int8/int4, sparsity, and model parallelism are not measured runtime executions.             The project should present these as planning estimates, not deployment benchmarks.

## Main Interpretation

Model parallelism reduces per-device memory by partitioning model parameters across devices.

Total-memory reduction is lower than parameter-only reduction because non-parameter memory is assumed replicated.

Quantization, sparsity, and model parallelism solve memory pressure in different ways.

These results are analytical and not measured distributed runtime profiling.

## Next Step

Day 55 will summarize the model-parallel phase and finalize the optimization-technique comparison.

---

# Day 55 - Model-parallel Phase Summary and Final Optimization Comparison

## Goal

Summarize the model-parallel phase and create a final comparison across precision, quantization, sparsity, and model parallelism.

## Files Created

- results/model_parallel_phase_summary.csv
- results/final_optimization_comparison.csv
- results/final_optimization_recommendations.csv
- results/final_optimization_key_findings.csv
- report/model_parallel_phase_summary_report.md

## Model-parallel Phase Summary

   day                                phase_step                                                                                       main_output                                                                                                                                                   main_result                                                                scope
Day 52              Model-parallel memory theory                                          Analytical per-device parameter partitioning simulation.                            With 8% combined overhead, effective parameter-memory reduction was 42% for 2 devices, 67% for 4 devices, and 79.5% for 8 devices.                       Analytical parameter-memory partitioning only.
Day 53       ModelParallelMemoryEstimator module                       Reusable estimator under src/estimators/model_parallel_memory_estimator.py.                                           Estimator computes ideal partitioned memory, overhead memory, estimated per-device memory, and effective reduction.                                  Reusable analytical estimator code.
Day 54 Model-parallel total-memory approximation Combined measured fp32 runtime memory with analytical model-parallel per-device parameter memory. Average total-memory reduction: 38.88% for fp32 2-device, 62.03% for fp32 4-device, 73.60% for fp32 8-device, and 77.30% for fp16 4-device model parallelism. Analytical total-memory approximation, not real multi-GPU profiling.

## Final Optimization Comparison

 rank_by_avg_reduction                             method             optimization_type  avg_total_reduction_percent  distilgpt2_reduction_percent  gpt2_reduction_percent runtime_measured                        claim_status                                                                  main_tradeoff
                     1                  int4 quantization           weight quantization                    80.282851                     78.906128               81.659573               no            analytical estimate only     Largest analytical quantization saving; quality and kernel support matter.
                     2 fp16 model parallelism - 4 devices precision + model parallelism                    77.302983                     75.977360               78.628605               no            analytical estimate only              Combines dtype reduction and model partitioning; analytical only.
                     3          90% unstructured sparsity                sparse storage                    74.062738                     72.792680               75.332796               no            analytical estimate only     Strong analytical saving; real sparse runtime depends on kernels/hardware.
                     4 fp32 model parallelism - 8 devices             model parallelism                    73.599846                     72.337726               74.861966               no            analytical estimate only           Large per-device reduction but higher distributed-system complexity.
                     5                  int8 quantization           weight quantization                    68.710548                     67.532272               69.888824               no            analytical estimate only Strong analytical memory reduction; needs real quantized execution validation.
                     6 fp32 model parallelism - 4 devices             model parallelism                    62.027543                     60.963870               63.091216               no            analytical estimate only     Stronger per-device memory reduction; more communication and coordination.
                     7                     fp16 precision           precision reduction                    46.289211                     45.495425               47.082997  partly measured measured + analytical approximation                 Practical and widely supported; measured fp16 behavior exists.
                     8          75% unstructured sparsity                sparse storage                    46.289211                     45.495425               47.082997               no            analytical estimate only                  Useful only after sparsity overcomes index metadata overhead.
                     9 fp32 model parallelism - 2 devices             model parallelism                    38.882937                     38.216157               39.549718               no            analytical estimate only        Reduces per-device memory but needs multiple devices and communication.

## Recommendations

                                    use_case                                  recommended_method                                                                                                                   reason                                                                                   caution
Most defensible measured optimization result                                      fp16 precision                                                  fp32/fp16 runtime memory was actually measured for distilgpt2 and gpt2. The total-memory approximation is still model-based, but fp16 runtime measurements exist.
         Largest analytical memory reduction                                   int4 quantization                                          It gives the highest average analytical total-memory reduction at about 80.28%.                          No real int4 execution, latency, or quality impact was measured.
  Best multi-device memory planning strategy                   fp16 + 4-device model parallelism    It combines dtype reduction and parameter partitioning, giving about 77.30% estimated average total-memory reduction.                                                No real multi-GPU execution was performed.
       Best sparsity-based analytical result                           90% unstructured sparsity                                                          It gives about 74.06% estimated average total-memory reduction.              Real sparse runtime depends heavily on sparse kernels, format, and hardware.
               Safest way to present project Separate measured results from analytical estimates                                     fp16 was measured, while int8/int4, sparsity, and model parallelism were analytical.                                 Do not overclaim runtime behavior for analytical modules.
                      Next project direction                             Architecture comparison Optimization techniques are now complete enough. Next step is comparing CNN-style and Transformer-style memory behavior.                       Do not keep extending optimization theory without new measurements.

## Key Findings

 finding_id                                                                   finding                                                                                                                                           evidence                                                                                      interpretation
          1       Optimization techniques reduce memory through different mechanisms. Quantization reduces bytes per value, sparsity removes zero weights but adds metadata, and model parallelism partitions parameters across devices.                            They are not interchangeable; each solves a different memory constraint.
          2             The highest analytical reduction came from int4 quantization.                                                               int4 quantization ranked first with 80.28% average estimated total-memory reduction.         Low-bit quantization is highly memory-efficient analytically, but needs runtime validation.
          3            fp16 remains the most defensible measured optimization result.                                      fp16 precision gave 46.29% average estimated total-memory reduction and was based on measured fp32/fp16 runs.                           This should be presented as the strongest measured optimization evidence.
          4 Model parallelism becomes competitive when more devices or fp16 are used.                                  fp32 4-device model parallelism gave 62.03% average reduction, while fp16 4-device model parallelism gave 77.30%. Model parallelism is useful for per-device memory reduction but adds distributed-system complexity.
          5            High sparsity can compete with model parallelism analytically.                                                                    90% unstructured sparsity gave 74.06% average estimated total-memory reduction.                Sparse storage can be effective, but only after index metadata overhead is overcome.
          6                Measured and analytical results must be clearly separated.                                                   int8/int4 quantization, sparsity, and model parallelism were not measured as runtime executions.                                                       This keeps the project honest and defensible.

## Main Interpretation

The optimization-aware phase now covers precision, quantization, sparsity, and model parallelism.

fp16 is the strongest measured optimization result.

int4 quantization gives the highest analytical memory reduction.

90% sparsity and model parallelism also give strong analytical per-device memory reduction, but both need runtime validation.

Measured and analytical results must be clearly separated.

## Next Step

Day 56 will start architecture comparison between CNN-style and Transformer-style workloads.

---

# Day 56 - Architecture Comparison Setup

## Goal

Start the architecture comparison phase by defining simple CNN and Transformer-style baselines.

## Files Created

- results/architecture_model_specs.csv
- results/architecture_parameter_memory_comparison.csv
- results/architecture_forward_memory_profile.csv
- results/architecture_initial_key_findings.csv
- report/architecture_comparison_setup_report.md

## Model Specs

architecture                  model_name     input_type     example_input_shape                                                      main_layers                                                             main_memory_drivers  attention_based
         CNN                   SimpleCNN          image   batch x 3 x 224 x 224               Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear                            image resolution, channels, feature maps, batch size            False
 Transformer SimpleTransformerClassifier token sequence batch x sequence_length Embedding, positional embedding, TransformerEncoderLayer, Linear sequence length, embedding size, attention heads, feed-forward size, batch size             True

## Parameter Comparison

architecture                  model_name  num_parameters  fp32_parameter_memory_MB  fp16_parameter_memory_MB                                                                                              main_interpretation
         CNN                   SimpleCNN           94538                  0.360634                  0.180317 Small CNN has relatively low parameter memory; activation memory can become important for high image resolution.
 Transformer SimpleTransformerClassifier        11040778                 42.117226                 21.058613            Transformer has higher parameter memory due to embeddings, attention layers, and feed-forward blocks.

## Forward Memory Profile

architecture model_name batch_size        input_shape sequence_length peak_allocated_MB peak_reserved_MB final_allocated_MB final_reserved_MB                                            scope_note
         N/A        N/A       None CUDA not available            None              None             None               None              None CUDA not available; forward memory profiling skipped.

## Key Findings

 finding_id                                                                      finding                                                                                                                                                                                         evidence                                                                                                why_it_matters
          1             CNN and Transformer architectures have different memory drivers. CNN memory depends heavily on image resolution, channels, feature maps, and batch size; Transformer memory depends on sequence length, embedding size, attention heads, and feed-forward blocks.        A single memory estimator may not generalize across architectures without architecture-aware features.
          2 The simple Transformer has much higher parameter memory than the simple CNN.                                                                                                            The Transformer fp32 parameter memory is about 116.79x the CNN fp32 parameter memory. Embedding layers and Transformer blocks can dominate parameter memory even in small Transformer-style models.
          3      Attention-based models require sequence-length-aware memory estimation.                                                                                                                   Transformer memory depends on token sequence length and attention computation.                     This supports the need for separate handling of sequence length in LLM memory estimation.
          4  CNN-style workloads need image-resolution and feature-map-aware estimation.                                                                                               CNN activations are spatial tensors whose size depends on height, width, channels, and batch size.                           Vision-style models require different memory features than text Transformer models.
          5                           Day 56 is a setup day for architecture comparison.                                                                         Simple CNN and Transformer models were defined, parameter memory was compared, and optional forward profiling was added.                              This creates the foundation for deeper architecture comparison in the next days.

## Main Interpretation

CNN and Transformer architectures have different memory drivers.

CNN memory depends on image resolution, channels, feature maps, and batch size.

Transformer memory depends on sequence length, embedding size, attention heads, feed-forward blocks, and batch size.

## Next Step

Day 57 will analyze CNN memory scaling with image resolution and batch size.

---

# Day 57 - CNN Memory Scaling

## Goal

Analyze how SimpleCNN memory changes with image resolution and batch size.

## Files Created

- results/cnn_resolution_batch_memory_profile.csv
- results/cnn_activation_memory_estimate.csv
- results/cnn_activation_memory_detail.csv
- results/cnn_memory_scaling_summary.csv
- results/cnn_memory_scaling_key_findings.csv
- plots/cnn_peak_memory_by_resolution.png
- plots/cnn_peak_memory_by_batch_size.png
- plots/cnn_activation_memory_estimate.png
- report/cnn_memory_scaling_report.md

## Activation Summary

 batch_size  image_size  estimated_total_activation_memory_MB                                                               scope_note
          1          64                              1.109863 Analytical activation-memory estimate for SimpleCNN forward activations.
          1         128                              4.437988 Analytical activation-memory estimate for SimpleCNN forward activations.
          1         224                             13.590332 Analytical activation-memory estimate for SimpleCNN forward activations.
          1         384                             39.937988 Analytical activation-memory estimate for SimpleCNN forward activations.
          2          64                              2.219727 Analytical activation-memory estimate for SimpleCNN forward activations.
          2         128                              8.875977 Analytical activation-memory estimate for SimpleCNN forward activations.
          2         224                             27.180664 Analytical activation-memory estimate for SimpleCNN forward activations.
          2         384                             79.875977 Analytical activation-memory estimate for SimpleCNN forward activations.
          4          64                              4.439453 Analytical activation-memory estimate for SimpleCNN forward activations.
          4         128                             17.751953 Analytical activation-memory estimate for SimpleCNN forward activations.
          4         224                             54.361328 Analytical activation-memory estimate for SimpleCNN forward activations.
          4         384                            159.751953 Analytical activation-memory estimate for SimpleCNN forward activations.
          8          64                              8.878906 Analytical activation-memory estimate for SimpleCNN forward activations.
          8         128                             35.503906 Analytical activation-memory estimate for SimpleCNN forward activations.
          8         224                            108.722656 Analytical activation-memory estimate for SimpleCNN forward activations.
          8         384                            319.503906 Analytical activation-memory estimate for SimpleCNN forward activations.

## Forward Profile

architecture model_name  batch_size  image_size       input_shape  num_parameters  fp32_parameter_memory_MB  analytical_activation_memory_MB  peak_allocated_MB  peak_reserved_MB  final_allocated_MB  final_reserved_MB profiling_status                                       scope_note
         CNN  SimpleCNN           1          64   1 x 3 x 64 x 64           94538                  0.360634                         1.109863           9.534668              22.0            9.534180               22.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           2          64   2 x 3 x 64 x 64           94538                  0.360634                         2.219727          11.580566              24.0            9.581055               24.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           4          64   4 x 3 x 64 x 64           94538                  0.360634                         4.439453          13.674316              26.0            9.674805               26.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           8          64   8 x 3 x 64 x 64           94538                  0.360634                         8.878906          17.861816              26.0            9.862305               26.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           1         128 1 x 3 x 128 x 128           94538                  0.360634                         4.437988          13.674316              26.0            9.674805               26.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           2         128 2 x 3 x 128 x 128           94538                  0.360634                         8.875977          17.861816              26.0            9.862305               26.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           4         128 4 x 3 x 128 x 128           94538                  0.360634                        17.751953          26.236816              46.0           10.237305               46.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           8         128 8 x 3 x 128 x 128           94538                  0.360634                        35.503906          42.986816              56.0           10.987305               56.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           1         224 1 x 3 x 224 x 224           94538                  0.360634                        13.590332          22.311035              44.0           10.061523               44.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           2         224 2 x 3 x 224 x 224           94538                  0.360634                        27.180664          35.135254              52.0           10.635742               52.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           4         224 4 x 3 x 224 x 224           94538                  0.360634                        54.361328          60.783691              76.0           11.784180               76.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           8         224 8 x 3 x 224 x 224           94538                  0.360634                       108.722656         114.080566             124.0           14.081055              124.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           1         384 1 x 3 x 384 x 384           94538                  0.360634                        39.937988          47.174316              60.0           11.174805               60.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           2         384 2 x 3 x 384 x 384           94538                  0.360634                        79.875977          84.861816              96.0           12.862305               96.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           4         384 4 x 3 x 384 x 384           94538                  0.360634                       159.751953         160.236816             168.0           16.237305              168.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.
         CNN  SimpleCNN           8         384 8 x 3 x 384 x 384           94538                  0.360634                       319.503906         311.486816             326.0           23.487305              326.0    measured_cuda Measured CUDA forward-pass memory for SimpleCNN.

## Scaling Summary

scaling_dimension    fixed_value  min_batch_size  max_batch_size  activation_memory_at_min_batch_MB  activation_memory_at_max_batch_MB                                                                                    scaling_interpretation  min_image_size  max_image_size  activation_memory_at_min_resolution_MB  activation_memory_at_max_resolution_MB
       batch_size  image_size=64             1.0             8.0                           1.109863                           8.878906                                      CNN activation memory scales approximately linearly with batch size.             NaN             NaN                                     NaN                                     NaN
       batch_size image_size=128             1.0             8.0                           4.437988                          35.503906                                      CNN activation memory scales approximately linearly with batch size.             NaN             NaN                                     NaN                                     NaN
       batch_size image_size=224             1.0             8.0                          13.590332                         108.722656                                      CNN activation memory scales approximately linearly with batch size.             NaN             NaN                                     NaN                                     NaN
       batch_size image_size=384             1.0             8.0                          39.937988                         319.503906                                      CNN activation memory scales approximately linearly with batch size.             NaN             NaN                                     NaN                                     NaN
 image_resolution   batch_size=1             NaN             NaN                                NaN                                NaN CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W.            64.0           384.0                                1.109863                               39.937988
 image_resolution   batch_size=2             NaN             NaN                                NaN                                NaN CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W.            64.0           384.0                                2.219727                               79.875977
 image_resolution   batch_size=4             NaN             NaN                                NaN                                NaN CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W.            64.0           384.0                                4.439453                              159.751953
 image_resolution   batch_size=8             NaN             NaN                                NaN                                NaN CNN activation memory grows strongly with image resolution because spatial feature maps scale with H x W.            64.0           384.0                                8.878906                              319.503906

## Key Findings

 finding_id                                                                          finding                                                                                                         evidence                                                                                      why_it_matters
          1             CNN activation memory scales approximately linearly with batch size.            At image size 224, increasing batch size from 1 to 8 increased analytical activation memory by 8.00x.                                     Batch size is a direct memory-scaling factor for CNN workloads.
          2                      CNN activation memory grows strongly with image resolution.          At batch size 1, increasing image size from 64 to 384 increased analytical activation memory by 35.98x. CNN feature maps depend on spatial dimensions, so higher image resolution increases memory sharply.
          3 CNN parameter memory is small compared with activation memory for larger inputs. SimpleCNN fp32 parameter memory is only 0.3606 MB, while activation memory grows with image size and batch size.              For CNN inference, activations can dominate memory even when parameter count is small.
          4                     CNN memory estimation needs image-resolution-aware features.                                                     Activation memory changes with H x W feature-map dimensions.                        A GPT-style sequence-length estimator is not enough for CNN-style workloads.
          5                                         Day 57 focuses on CNN-side scaling only.                        Transformer scaling will be analyzed separately in the next architecture-comparison step.                                Separating CNN and Transformer scaling makes the comparison cleaner.

## Main Interpretation

CNN memory scales approximately linearly with batch size.

CNN memory grows strongly with image resolution because feature maps depend on H x W.

SimpleCNN has very small parameter memory, but activation memory grows with input size.

## Next Step

Day 58 will analyze Transformer memory scaling with sequence length and batch size.

---