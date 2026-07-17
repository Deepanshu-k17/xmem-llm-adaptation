from dataclasses import dataclass
from typing import Optional

from src.estimators.combined_inference_estimator import CombinedInferenceEstimator
from src.estimators.training_memory_estimator_v4 import (
    TrainingMemoryEstimatorV4,
)


@dataclass
class RecommendationResult:
    """
    Stores the final recommendation returned by the engine.
    """

    fits: bool
    estimated_memory_mb: float
    available_memory_mb: float
    recommended_precision: str
    recommended_batch_size: int
    recommended_quantization: Optional[str]
    cpu_offloading: bool
    gradient_checkpointing: bool
    safety_margin_percent: float
    reason: str


class RecommendationEngine:
    """
    Unified recommendation engine.

    Combines all existing memory estimators and recommends
    the best configuration for inference or training.
    """

    def __init__(
        self,
        inference_estimator=None,
        training_estimator=None,
        safety_margin=7.5,
    ):

        self.inference_estimator = (
            inference_estimator
            if inference_estimator
            else CombinedInferenceEstimator()
        )

        self.training_estimator = (
            training_estimator
            if training_estimator
            else TrainingMemoryEstimatorV4()
        )

        self.safety_margin = safety_margin

    ####################################################################
    # Helper Functions
    ####################################################################

    def _gpu_memory_mb(self, gpu_memory_gb):
        return gpu_memory_gb * 1024

    def _apply_safety_margin(self, memory_mb):
        return memory_mb * (
            1 + self.safety_margin / 100
        )

    def _fits(self, required_mb, available_mb):
        return required_mb <= available_mb

    ####################################################################
    # Inference Recommendation
    ####################################################################

    def recommend_inference(
        self,
        model_name,
        gpu_memory_gb,
        batch_size,
        input_tokens,
        max_new_tokens,
        dtype="fp32",
        use_cache=True,
    ):

        estimate = self.inference_estimator.estimate(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            use_cache=use_cache,
        )

        estimated_memory = estimate[
            "predicted_peak_reserved_MB"
        ]

        estimated_memory = self._apply_safety_margin(
            estimated_memory
        )

        available_memory = self._gpu_memory_mb(
            gpu_memory_gb
        )

        if self._fits(
            estimated_memory,
            available_memory,
        ):

            return RecommendationResult(
                fits=True,
                estimated_memory_mb=estimated_memory,
                available_memory_mb=available_memory,
                recommended_precision=dtype,
                recommended_batch_size=batch_size,
                recommended_quantization=None,
                cpu_offloading=False,
                gradient_checkpointing=False,
                safety_margin_percent=self.safety_margin,
                reason="Model comfortably fits into GPU memory.",
            )

        ###############################################################
        # Doesn't Fit
        ###############################################################

        reason = []
        recommended_precision = dtype
        recommended_batch_size = batch_size
        cpu_offloading = False
        quantization = None

        if batch_size > 1:
            recommended_batch_size = max(1, batch_size // 2)
            reason.append("Reduce batch size.")

        if dtype == "fp32":
            recommended_precision = "fp16"
            reason.append("Switch to FP16.")
        else:
            quantization = "int8"
            reason.append("Consider INT8 quantization.")

        if estimated_memory > available_memory * 1.5:
            cpu_offloading = True
            reason.append("CPU offloading recommended.")

        return RecommendationResult(
            fits=False,
            estimated_memory_mb=estimated_memory,
            available_memory_mb=available_memory,
            recommended_precision=recommended_precision,
            recommended_batch_size=recommended_batch_size,
            recommended_quantization=quantization,
            cpu_offloading=cpu_offloading,
            gradient_checkpointing=False,
            safety_margin_percent=self.safety_margin,
            reason=" ".join(reason),
        )
