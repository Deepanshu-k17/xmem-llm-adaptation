
from estimators.improved_base_estimator import ImprovedBaseMemoryEstimator
from estimators.kv_cache_estimator import KVCacheEstimator
from estimators.precision_estimator import PrecisionAwareEstimator
from estimators.allocator_correction import AllocatorCorrectionEstimator


class CombinedInferenceEstimator:
    def __init__(
        self,
        kv_weight=0.5,
        allocator_padding_ratio=0.08540076335877858,
        allocator_min_padding_mb=8.0,
        allocator_rounding_mb=2.0,
    ):
        self.kv_weight = kv_weight

        self.base_estimator = ImprovedBaseMemoryEstimator()
        self.kv_estimator = KVCacheEstimator()
        self.precision_estimator = PrecisionAwareEstimator()
        self.allocator_estimator = AllocatorCorrectionEstimator(
            default_padding_ratio=allocator_padding_ratio,
            min_padding_mb=allocator_min_padding_mb,
            rounding_mb=allocator_rounding_mb,
        )

    def estimate(
        self,
        model_name,
        batch_size,
        input_tokens,
        max_new_tokens,
        dtype="fp32",
        use_cache=True,
    ):
        dtype = str(dtype).lower()

        base_result = self.base_estimator.estimate_peak_allocated_mb(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            dtype="fp32",
            use_cache=use_cache,
        )

        kv_result = self.kv_estimator.estimate_kv_cache_mb(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            dtype="fp32",
            use_cache=use_cache,
        )

        fp32_allocated_prediction = (
            base_result["predicted_peak_allocated_MB"]
            + self.kv_weight * kv_result["kv_cache_MB"]
        )

        precision_result = self.precision_estimator.estimate_precision_adjusted_memory(
            fp32_reference_memory_mb=fp32_allocated_prediction,
            dtype=dtype,
        )

        predicted_allocated = precision_result["precision_adjusted_MB"]

        reserved_result = self.allocator_estimator.estimate_reserved_memory_mb(
            predicted_allocated_mb=predicted_allocated
        )

        return {
            "predicted_peak_allocated_MB": predicted_allocated,
            "predicted_peak_reserved_MB": reserved_result["predicted_reserved_MB"],
            "fp32_allocated_prediction_MB": fp32_allocated_prediction,
            "base_prediction_MB": base_result["predicted_peak_allocated_MB"],
            "estimated_kv_cache_MB": kv_result["kv_cache_MB"],
            "kv_weight": self.kv_weight,
            "dtype_scale": precision_result["dtype_scale"],
            "precision_correction": precision_result.get("precision_correction", 1.0),
            "allocator_padding_MB": reserved_result["allocator_padding_MB"],
            "allocator_padding_ratio": reserved_result["padding_ratio"],
            "parameter_memory_MB": base_result["parameter_memory_MB"],
            "token_memory_component_MB": base_result["token_memory_component_MB"],
            "batch_memory_component_MB": base_result["batch_memory_component_MB"],
            "model_size_overhead_MB": base_result["model_size_overhead_MB"],
            "total_tokens": base_result["total_tokens"],
        }
