
from estimators.model_config_utils import (
    estimate_parameter_memory_mb,
    get_model_config_values,
    get_dtype_bytes,
)


class ImprovedBaseMemoryEstimator:
    def __init__(
        self,
        token_factor=0.002,
        batch_factor=2.0,
        overhead_ratio=0.06,
        min_overhead_mb=1.0,
        max_overhead_mb=25.0,
    ):
        self.token_factor = token_factor
        self.batch_factor = batch_factor
        self.overhead_ratio = overhead_ratio
        self.min_overhead_mb = min_overhead_mb
        self.max_overhead_mb = max_overhead_mb

    def get_model_size_overhead(self, parameter_memory_mb):
        overhead = parameter_memory_mb * self.overhead_ratio

        if overhead < self.min_overhead_mb:
            overhead = self.min_overhead_mb

        if overhead > self.max_overhead_mb:
            overhead = self.max_overhead_mb

        return overhead

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
        total_tokens = int(input_tokens) + int(max_new_tokens)

        dtype_bytes = get_dtype_bytes(dtype)
        hidden_size = config_info["hidden_size"] or 1
        num_layers = config_info["num_layers"] or 1

        token_memory_mb = (
            int(batch_size)
            * total_tokens
            * hidden_size
            * num_layers
            * dtype_bytes
            * self.token_factor
            / (1024 ** 2)
        )

        batch_memory_mb = int(batch_size) * self.batch_factor

        overhead_mb = self.get_model_size_overhead(parameter_memory_mb)

        cache_multiplier = 1.0
        if bool(use_cache):
            cache_multiplier = 1.02

        predicted = (
            parameter_memory_mb
            + token_memory_mb
            + batch_memory_mb
            + overhead_mb
        )

        predicted = predicted * cache_multiplier

        return {
            "predicted_peak_allocated_MB": predicted,
            "parameter_memory_MB": parameter_memory_mb,
            "token_memory_component_MB": token_memory_mb,
            "batch_memory_component_MB": batch_memory_mb,
            "model_size_overhead_MB": overhead_mb,
            "total_tokens": total_tokens,
        }
