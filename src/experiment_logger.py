
import os
import csv
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CSV_FIELDS = [
    "timestamp",
    "platform",
    "gpu_name",
    "model_name",
    "mode",
    "batch_size",
    "input_tokens",
    "max_new_tokens",
    "output_tokens",
    "dtype",
    "use_cache",
    "peak_allocated_MB",
    "peak_reserved_MB",
    "final_allocated_MB",
    "final_reserved_MB",
    "runtime_sec",
    "oom",
    "error_message"
]


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_stats():
    if not torch.cuda.is_available():
        return {
            "peak_allocated_MB": 0,
            "peak_reserved_MB": 0,
            "final_allocated_MB": 0,
            "final_reserved_MB": 0,
        }

    return {
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "final_allocated_MB": torch.cuda.memory_allocated() / (1024 ** 2),
        "final_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2),
    }


def append_to_csv(row, path):
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def make_input_ids(tokenizer, input_tokens, batch_size, device):
    base_ids = tokenizer.encode(
        "Hello world, this is a memory profiling experiment.",
        add_special_tokens=False
    )

    repeated = []
    while len(repeated) < input_tokens:
        repeated.extend(base_ids)

    repeated = repeated[:input_tokens]

    input_ids = torch.tensor([repeated] * batch_size, dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def run_inference_experiment(
    results_path,
    platform="colab",
    model_name="sshleifer/tiny-gpt2",
    input_tokens=32,
    max_new_tokens=32,
    batch_size=1,
    dtype="fp32",
    use_cache=True,
):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    row = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform,
        "gpu_name": gpu_name,
        "model_name": model_name,
        "mode": "inference",
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "max_new_tokens": max_new_tokens,
        "output_tokens": 0,
        "dtype": dtype,
        "use_cache": use_cache,
        "peak_allocated_MB": 0,
        "peak_reserved_MB": 0,
        "final_allocated_MB": 0,
        "final_reserved_MB": 0,
        "runtime_sec": 0,
        "oom": False,
        "error_message": "",
    }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        clear_gpu_memory()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)

        model.eval()

        inputs = make_input_ids(
            tokenizer=tokenizer,
            input_tokens=input_tokens,
            batch_size=batch_size,
            device=device
        )

        clear_gpu_memory()

        start = time.time()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id
            )

        end = time.time()

        mem = get_memory_stats()

        row["output_tokens"] = int(output.shape[1])
        row["peak_allocated_MB"] = round(mem["peak_allocated_MB"], 2)
        row["peak_reserved_MB"] = round(mem["peak_reserved_MB"], 2)
        row["final_allocated_MB"] = round(mem["final_allocated_MB"], 2)
        row["final_reserved_MB"] = round(mem["final_reserved_MB"], 2)
        row["runtime_sec"] = round(end - start, 3)

        del model
        del tokenizer
        del inputs
        del output
        clear_gpu_memory()

    except RuntimeError as e:
        row["oom"] = "out of memory" in str(e).lower()
        row["error_message"] = str(e)[:300]
        clear_gpu_memory()

    except Exception as e:
        row["error_message"] = str(e)[:300]
        clear_gpu_memory()

    append_to_csv(row, results_path)
    return row
