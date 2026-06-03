
from estimators.model_config_utils import (
    estimate_parameter_memory_mb,
    get_model_config_values,
    get_dtype_bytes,
)


class BaseMemoryEstimator:
    def __init__(
        self,
        token_factor=0.002,
        batch_factor=2.0,
        overhead_factor=1.05,
        overhead_constant_mb=20.0,
    ):
        self.token_factor = token_factor
        self.batch_factor = batch_factor
        self.overhead_factor = overhead_factor
        self.overhead_constant_mb = overhead_constant_mb

    def estimate_peak_allocated_mb(
        self,
        model_name,
        batch_size,
        input_tokens,
        max_new_tokens,
        dtype="fp32",
        use_cache=True,
    ):
        param_info = estimate_parameter_memory_mb(model_name, dtype=dtype)
        config_info = get_model_config_values(model_name)

        parameter_memory_mb = param_info["parameter_memory_MB"]
        total_tokens = input_tokens + max_new_tokens
        dtype_bytes = get_dtype_bytes(dtype)

        hidden_size = config_info["hidden_size"] or 1
        num_layers = config_info["num_layers"] or 1

        token_memory_mb = (
            batch_size
            * total_tokens
            * hidden_size
            * num_layers
            * dtype_bytes
            * self.token_factor
            / (1024 ** 2)
        )

        batch_memory_mb = batch_size * self.batch_factor

        cache_multiplier = 1.0
        if use_cache:
            cache_multiplier = 1.02

        predicted = (
            parameter_memory_mb
            + token_memory_mb
            + batch_memory_mb
            + self.overhead_constant_mb
        )

        predicted = predicted * self.overhead_factor * cache_multiplier

        return {
            "predicted_peak_allocated_MB": predicted,
            "parameter_memory_MB": parameter_memory_mb,
            "token_memory_component_MB": token_memory_mb,
            "batch_memory_component_MB": batch_memory_mb,
            "total_tokens": total_tokens,
        }
