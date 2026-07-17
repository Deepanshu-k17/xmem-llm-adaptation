"""
decision_engine.py

Decision engine for evaluating whether a predicted memory requirement
fits within the available GPU memory.
"""


class DecisionEngine:
    """
    Evaluates memory predictions against available GPU memory.

    Status Levels:
        SAFE     : Utilization < warning threshold
        WARNING  : Warning threshold <= Utilization <= danger threshold
        OOM      : Utilization > danger threshold
    """

    def __init__(self, warning_threshold=0.90, danger_threshold=1.00):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold

    def evaluate(self, predicted_memory_mb, gpu_memory_mb):
        """
        Evaluate whether the predicted memory fits into the GPU.
        """

        if gpu_memory_mb <= 0:
            raise ValueError("GPU memory must be greater than zero.")

        recommendations = []

        utilization = predicted_memory_mb / gpu_memory_mb
        utilization_percent = utilization * 100
        remaining_memory_mb = gpu_memory_mb - predicted_memory_mb

        if utilization < self.warning_threshold:
            status = "SAFE"

            recommendations.append(
                "Training should fit comfortably within GPU memory."
            )
            recommendations.append(
                "No optimization is required."
            )

        elif utilization <= self.danger_threshold:
            status = "WARNING"

            recommendations.append(
                "GPU memory usage is high."
            )
            recommendations.append(
                "Consider reducing the batch size."
            )
            recommendations.append(
                "Mixed precision training is recommended."
            )

        else:
            status = "OOM"

            recommendations.append(
                "Predicted memory exceeds available GPU memory."
            )
            recommendations.append(
                "Reduce the batch size."
            )
            recommendations.append(
                "Enable mixed precision."
            )
            recommendations.append(
                "Consider gradient checkpointing."
            )

        return {
            "status": status,
            "utilization": utilization,
            "utilization_percent": utilization_percent,
            "predicted_memory_mb": predicted_memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "remaining_memory_mb": remaining_memory_mb,
            "recommendations": recommendations,
        }
