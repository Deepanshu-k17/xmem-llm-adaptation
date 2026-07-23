
import os
import time
import gc
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_memory_snapshot(stage_name):
    if not torch.cuda.is_available():
        return {
            "stage": stage_name,
            "allocated_MB": 0.0,
            "reserved_MB": 0.0,
            "peak_allocated_MB": 0.0,
            "peak_reserved_MB": 0.0,
        }

    torch.cuda.synchronize()

    return {
        "stage": stage_name,
        "allocated_MB": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_MB": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "peak_allocated_MB": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }


def make_dummy_text(batch_size, input_tokens):
    base_words = [
        "memory", "prediction", "for", "large", "language", "models",
        "requires", "profiling", "training", "and", "inference", "workloads"
    ]

    text = " ".join(base_words)

    repeated = []
    for _ in range(batch_size):
        repeated.append((text + " ") * max(1, input_tokens // len(base_words) + 2))

    return repeated


def append_training_result(csv_path, result_row):
    df = pd.DataFrame([result_row])

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(csv_path, index=False)


def run_training_memory_experiment(
    model_name,
    batch_size,
    input_tokens,
    dtype="fp32",
    optimizer_name="adamw",
    learning_rate=5e-5,
    device_name="colab",
    csv_path=None,
):
    result = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "device_name": device_name,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_name": model_name,
        "task": "training",
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "dtype": dtype,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "oom": False,
        "error_message": "",
    }

    stage_rows = []

    try:
        cleanup_memory()
        start_time = time.time()

        stage_rows.append(get_memory_snapshot("before_model_load"))

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if dtype == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()

        stage_rows.append(get_memory_snapshot("after_model_load"))

        texts = make_dummy_text(batch_size, input_tokens)

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=input_tokens,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        labels = input_ids.clone()

        stage_rows.append(get_memory_snapshot("after_batch_creation"))

        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer_name. Use 'sgd' or 'adamw'.")

        stage_rows.append(get_memory_snapshot("after_optimizer_creation"))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        stage_rows.append(get_memory_snapshot("after_forward"))

        loss = outputs.loss

        stage_rows.append(get_memory_snapshot("after_loss"))

        loss.backward()

        stage_rows.append(get_memory_snapshot("after_backward"))

        optimizer.step()

        stage_rows.append(get_memory_snapshot("after_optimizer_step"))

        optimizer.zero_grad(set_to_none=True)

        stage_rows.append(get_memory_snapshot("after_zero_grad"))

        runtime_sec = round(time.time() - start_time, 3)

        final_snapshot = get_memory_snapshot("final")

        result.update({
            "loss": round(float(loss.detach().cpu()), 6),
            "runtime_sec": runtime_sec,
            "final_allocated_MB": final_snapshot["allocated_MB"],
            "final_reserved_MB": final_snapshot["reserved_MB"],
            "peak_allocated_MB": final_snapshot["peak_allocated_MB"],
            "peak_reserved_MB": final_snapshot["peak_reserved_MB"],
        })

        del outputs, loss, optimizer, model, input_ids, attention_mask, labels, encoded
        cleanup_memory()

    except torch.cuda.OutOfMemoryError as e:
        result["oom"] = True
        result["error_message"] = str(e)[:300]

        final_snapshot = get_memory_snapshot("oom_final")
        result.update({
            "loss": None,
            "runtime_sec": None,
            "final_allocated_MB": final_snapshot["allocated_MB"],
            "final_reserved_MB": final_snapshot["reserved_MB"],
            "peak_allocated_MB": final_snapshot["peak_allocated_MB"],
            "peak_reserved_MB": final_snapshot["peak_reserved_MB"],
        })

        cleanup_memory()

    except Exception as e:
        result["oom"] = False
        result["error_message"] = str(e)[:300]

        final_snapshot = get_memory_snapshot("error_final")
        result.update({
            "loss": None,
            "runtime_sec": None,
            "final_allocated_MB": final_snapshot["allocated_MB"],
            "final_reserved_MB": final_snapshot["reserved_MB"],
            "peak_allocated_MB": final_snapshot["peak_allocated_MB"],
            "peak_reserved_MB": final_snapshot["peak_reserved_MB"],
        })

        cleanup_memory()

    stage_df = pd.DataFrame(stage_rows)

    if csv_path is not None:
        append_training_result(csv_path, result)

    return result, stage_df
