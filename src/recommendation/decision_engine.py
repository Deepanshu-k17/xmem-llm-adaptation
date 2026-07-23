"""
decision_engine.py
"""

from src.recommendation.batch_size_recommender import BatchSizeRecommender


class DecisionEngine:

    def __init__(self, warning_threshold=0.90, danger_threshold=1.00):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold

    def evaluate(
        self,
        predicted_memory_mb,
        gpu_memory_mb,
        current_batch_size=None,
    ):

        if gpu_memory_mb <= 0:
            raise ValueError("GPU memory must be greater than zero.")

        utilization = predicted_memory_mb / gpu_memory_mb
        utilization_percent = utilization * 100
        remaining_memory_mb = gpu_memory_mb - predicted_memory_mb

        recommendations = []

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

            if current_batch_size is not None:

                batch = BatchSizeRecommender.recommend(
                    current_batch_size,
                    predicted_memory_mb,
                    gpu_memory_mb,
                )

                recommendations.append(
                    f"Reduce batch size from {batch['current_batch_size']} "
                    f"to {batch['recommended_batch_size']}."
                )

            recommendations.append(
                "Mixed precision training is recommended."
            )

        else:

            status = "OOM"

            recommendations.append(
                "Predicted memory exceeds available GPU memory."
            )

            if current_batch_size is not None:

                batch = BatchSizeRecommender.recommend(
                    current_batch_size,
                    predicted_memory_mb,
                    gpu_memory_mb,
                )

                recommendations.append(
                    f"Reduce batch size from {batch['current_batch_size']} "
                    f"to {batch['recommended_batch_size']}."
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
