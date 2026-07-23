from dataclasses import dataclass

from src.estimators.combined_inference_estimator import (
    CombinedInferenceEstimator,
)
from src.estimators.training_memory_estimator import (
    TrainingMemoryEstimator,
)


@dataclass
class RuntimeResult:
    """
    Runtime memory analysis result.
    """

    task: str
    estimated_reserved_memory_mb: float
    available_gpu_memory_mb: float
    safety_margin_percent: float
    required_memory_with_margin_mb: float
    fits: bool
    utilization_percent: float
    message: str


class RuntimeAdvisor:
    """
    Runtime GPU memory advisor.

    Uses:
    - CombinedInferenceEstimator
    - TrainingMemoryEstimator
    """

    def __init__(
        self,
        inference_estimator=None,
        training_estimator=None,
        safety_margin_percent=7.5,
    ):

        self.inference_estimator = (
            inference_estimator
            if inference_estimator is not None
            else CombinedInferenceEstimator()
        )

        self.training_estimator = (
            training_estimator
            if training_estimator is not None
            else TrainingMemoryEstimator()
        )

        self.safety_margin_percent = safety_margin_percent

    ####################################################################

    @staticmethod
    def gpu_memory_mb(gpu_memory_gb):
        return gpu_memory_gb * 1024

    ####################################################################

    def apply_safety_margin(self, memory_mb):
        return memory_mb * (
            1 + self.safety_margin_percent / 100
        )

    ####################################################################

    def analyze_inference(
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

        reserved_memory = estimate[
            "predicted_peak_reserved_MB"
        ]

        required_memory = self.apply_safety_margin(
            reserved_memory
        )

        available_memory = self.gpu_memory_mb(
            gpu_memory_gb
        )

        fits = required_memory <= available_memory

        utilization = (
            required_memory
            / available_memory
            * 100
        )

        return RuntimeResult(
            task="Inference",
            estimated_reserved_memory_mb=reserved_memory,
            available_gpu_memory_mb=available_memory,
            safety_margin_percent=self.safety_margin_percent,
            required_memory_with_margin_mb=required_memory,
            fits=fits,
            utilization_percent=utilization,
            message=(
                "Inference workload fits GPU memory."
                if fits
                else "Inference workload exceeds GPU memory."
            ),
        )

    ####################################################################

    def analyze_training(
        self,
        model_name,
        gpu_memory_gb,
        batch_size,
        input_tokens,
        optimizer_name,
        dtype="fp32",
    ):

        estimate = self.training_estimator.estimate_training_memory_mb(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
            optimizer_name=optimizer_name,
            dtype=dtype,
        )

        reserved_memory = estimate[
            "predicted_peak_reserved_MB"
        ]

        required_memory = self.apply_safety_margin(
            reserved_memory
        )

        available_memory = self.gpu_memory_mb(
            gpu_memory_gb
        )

        fits = required_memory <= available_memory

        utilization = (
            required_memory
            / available_memory
            * 100
        )

        return RuntimeResult(
            task="Training",
            estimated_reserved_memory_mb=reserved_memory,
            available_gpu_memory_mb=available_memory,
            safety_margin_percent=self.safety_margin_percent,
            required_memory_with_margin_mb=required_memory,
            fits=fits,
            utilization_percent=utilization,
            message=(
                "Training workload fits GPU memory."
                if fits
                else "Training workload exceeds GPU memory."
            ),
        )
