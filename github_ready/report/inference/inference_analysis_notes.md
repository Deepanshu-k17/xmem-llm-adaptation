
# Inference Analysis Notes - After Day 10

## Current Status

The inference profiling phase has collected memory data for tiny-gpt2 and distilgpt2.

Experiments completed so far:

1. tiny-gpt2 input token length variation
2. tiny-gpt2 generated token length variation
3. tiny-gpt2 use_cache True vs False
4. distilgpt2 input token length variation
5. distilgpt2 generated token length variation
6. distilgpt2 use_cache True vs False
7. distilgpt2 batch size variation
8. distilgpt2 fp32 vs fp16 precision comparison

## Key Findings So Far

### 1. tiny-gpt2 is useful only for pipeline validation

tiny-gpt2 showed almost flat memory behavior across input length, generated length, and cache settings.

It is too small to reveal strong LLM memory-scaling trends.

### 2. distilgpt2 shows meaningful memory scaling

distilgpt2 used much more memory than tiny-gpt2.

At 512 input tokens, tiny-gpt2 used 12.53 MB peak allocated memory, while distilgpt2 used 384.01 MB.

This was about 30.6 times more memory.

### 3. Input token length affects memory

For distilgpt2, peak allocated memory increased from 331.22 MB at 16 input tokens to 384.01 MB at 512 input tokens.

This was an increase of 52.79 MB, or about 15.9%.

### 4. Generated token length affects memory

For distilgpt2, peak allocated memory increased from 335.36 MB at 8 generated tokens to 341.35 MB at 256 generated tokens.

The increase was smaller than input-length scaling, but still visible.

### 5. Cache behavior is nuanced

For distilgpt2, use_cache=False used more peak allocated and reserved memory than use_cache=True under the tested setting.

This shows that memory behavior cannot be predicted by simple assumptions.

### 6. Batch size scaling is sub-linear

For distilgpt2, batch size increased from 1 to 8, but peak allocated memory increased only from 335.36 MB to 384.00 MB.

This happened because model weights are shared across the batch.

### 7. fp16 significantly reduces memory

fp16 reduced average peak allocated memory from 343.41 MB to 181.25 MB.

This was a 47.22% reduction.

## Why This Matters

The data shows that LLM memory depends on:

- model size
- input tokens
- generated tokens
- batch size
- dtype
- cache behavior
- allocator behavior

This supports the need for estimator modules:

- BaseMemoryEstimator
- KVCacheEstimator
- PrecisionAwareEstimator
- AllocatorCorrectionEstimator

## Next Step

The next phase should start building prediction modules.

The first module should be a simple BaseMemoryEstimator.
