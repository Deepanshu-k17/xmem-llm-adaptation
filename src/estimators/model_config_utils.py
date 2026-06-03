
from transformers import AutoConfig, AutoModelForCausalLM


def get_dtype_bytes(dtype):
    dtype = str(dtype).lower()

    if dtype == "fp32":
        return 4
    if dtype == "float32":
        return 4
    if dtype == "fp16":
        return 2
    if dtype == "float16":
        return 2
    if dtype == "bf16":
        return 2
    if dtype == "bfloat16":
        return 2
    if dtype == "int8":
        return 1

    return 4


def get_model_config_values(model_name):
    config = AutoConfig.from_pretrained(model_name)

    num_layers = (
        getattr(config, "n_layer", None)
        or getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )

    hidden_size = (
        getattr(config, "n_embd", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "d_model", None)
    )

    num_heads = (
        getattr(config, "n_head", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(config, "num_heads", None)
    )

    vocab_size = getattr(config, "vocab_size", None)

    return {
        "model_name": model_name,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "vocab_size": vocab_size,
    }


def get_parameter_count(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    param_count = sum(p.numel() for p in model.parameters())
    del model
    return param_count


def estimate_parameter_memory_mb(model_name, dtype="fp32"):
    param_count = get_parameter_count(model_name)
    bytes_per_param = get_dtype_bytes(dtype)
    memory_mb = param_count * bytes_per_param / (1024 ** 2)

    return {
        "model_name": model_name,
        "dtype": dtype,
        "param_count": param_count,
        "bytes_per_param": bytes_per_param,
        "parameter_memory_MB": memory_mb,
    }
