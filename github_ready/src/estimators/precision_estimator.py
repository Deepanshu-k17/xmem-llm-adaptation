
from estimators.model_config_utils import get_dtype_bytes


class PrecisionAwareEstimator:
    def __init__(
        self,
        non_scaling_memory_ratio=0.06,
        fp16_empirical_correction=0.94,
    ):
        self.non_scaling_memory_ratio = non_scaling_memory_ratio
        self.fp16_empirical_correction = fp16_empirical_correction

    def get_dtype_scale(self, dtype):
        dtype = str(dtype).lower()

        if dtype in ["fp32", "float32"]:
            return 1.0

        if dtype in ["fp16", "float16", "bf16", "bfloat16"]:
            return 0.5

        if dtype in ["int8"]:
            return 0.25

        return 1.0

    def estimate_precision_adjusted_memory(
        self,
        fp32_reference_memory_mb,
        dtype="fp32",
    ):
        dtype = str(dtype).lower()
        dtype_scale = self.get_dtype_scale(dtype)

        if dtype in ["fp32", "float32"]:
            return {
                "precision_adjusted_MB": fp32_reference_memory_mb,
                "dtype_scale": dtype_scale,
                "scaling_memory_MB": fp32_reference_memory_mb,
                "non_scaling_memory_MB": 0.0,
                "precision_correction": 1.0,
            }

        non_scaling_memory_mb = fp32_reference_memory_mb * self.non_scaling_memory_ratio
        scaling_memory_mb = fp32_reference_memory_mb - non_scaling_memory_mb

        adjusted = non_scaling_memory_mb + scaling_memory_mb * dtype_scale

        precision_correction = 1.0

        if dtype in ["fp16", "float16", "bf16", "bfloat16"]:
            precision_correction = self.fp16_empirical_correction
            adjusted = adjusted * precision_correction

        return {
            "precision_adjusted_MB": adjusted,
            "dtype_scale": dtype_scale,
            "scaling_memory_MB": scaling_memory_mb,
            "non_scaling_memory_MB": non_scaling_memory_mb,
            "precision_correction": precision_correction,
        }

    def estimate_dtype_memory_from_components(
        self,
        parameter_memory_mb,
        token_memory_mb,
        batch_memory_mb,
        overhead_mb,
        kv_cache_mb,
        dtype="fp32",
    ):
        dtype = str(dtype).lower()
        dtype_scale = self.get_dtype_scale(dtype)

        if dtype in ["fp32", "float32"]:
            return {
                "precision_adjusted_MB": (
                    parameter_memory_mb
                    + token_memory_mb
                    + batch_memory_mb
                    + overhead_mb
                    + kv_cache_mb
                ),
                "dtype_scale": dtype_scale,
            }

        scaled_memory = (
            parameter_memory_mb
            + token_memory_mb
            + kv_cache_mb
        ) * dtype_scale

        mostly_fixed_memory = batch_memory_mb + overhead_mb

        predicted = scaled_memory + mostly_fixed_memory

        if dtype in ["fp16", "float16", "bf16", "bfloat16"]:
            predicted = predicted * self.fp16_empirical_correction

        return {
            "precision_adjusted_MB": predicted,
            "dtype_scale": dtype_scale,
        }
