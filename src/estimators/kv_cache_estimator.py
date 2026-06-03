
from estimators.model_config_utils import (
    get_model_config_values,
    get_dtype_bytes,
)


class KVCacheEstimator:
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def estimate_kv_cache_mb(
        self,
        model_name,
        batch_size,
        input_tokens,
        max_new_tokens,
        dtype="fp32",
        use_cache=True,
    ):
        if not bool(use_cache):
            return {
                "kv_cache_MB": 0.0,
                "num_layers": None,
                "hidden_size": None,
                "total_tokens": int(input_tokens) + int(max_new_tokens),
                "bytes_per_element": get_dtype_bytes(dtype),
            }

        config_info = get_model_config_values(model_name)

        num_layers = config_info["num_layers"] or 1
        hidden_size = config_info["hidden_size"] or 1
        bytes_per_element = get_dtype_bytes(dtype)

        total_tokens = int(input_tokens) + int(max_new_tokens)

        kv_bytes = (
            int(batch_size)
            * total_tokens
            * int(num_layers)
            * int(hidden_size)
            * 2
            * int(bytes_per_element)
        )

        kv_mb = kv_bytes / (1024 ** 2)
        kv_mb = kv_mb * self.scale_factor

        return {
            "kv_cache_MB": kv_mb,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "total_tokens": total_tokens,
            "bytes_per_element": bytes_per_element,
        }
