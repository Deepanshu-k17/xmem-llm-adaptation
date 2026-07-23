"""
recommendation_orchestrator.py

Combines multiple recommenders into a single
optimization plan.
"""

from src.recommendation.decision_engine import DecisionEngine
from src.recommendation.batch_size_recommender import BatchSizeRecommender
from src.recommendation.precision_recommender import PrecisionRecommender
from src.recommendation.quantization_recommender import QuantizationRecommender


class RecommendationOrchestrator:

    def __init__(self):

        self.decision_engine = DecisionEngine()

    def recommend(
        self,
        predicted_memory_mb,
        gpu_memory_mb,
        current_batch_size,
    ):

        decision = self.decision_engine.evaluate(
            predicted_memory_mb,
            gpu_memory_mb,
            current_batch_size,
        )

        precision = PrecisionRecommender.recommend(
            predicted_memory_mb,
            gpu_memory_mb,
        )

        quantization = QuantizationRecommender.recommend(
            predicted_memory_mb,
            gpu_memory_mb,
        )

        batch = BatchSizeRecommender.recommend(
            current_batch_size,
            predicted_memory_mb,
            gpu_memory_mb,
        )

        plan = []

        if decision["status"] == "SAFE":

            plan.append(
                "Current configuration is safe."
            )

        elif decision["status"] == "WARNING":

            plan.append(
                "GPU utilization is high."
            )

            plan.append(
                precision["recommendation"]
            )

        else:

            plan.append(
                "Predicted memory exceeds GPU memory."
            )

            plan.append(
                precision["recommendation"]
            )

            plan.append(
                quantization["recommendation"]
            )

            plan.append(
                f"Reduce batch size from "
                f"{batch['current_batch_size']} "
                f"to "
                f"{batch['recommended_batch_size']}."
            )

        return {

            "status": decision["status"],

            "predicted_memory_mb": predicted_memory_mb,

            "gpu_memory_mb": gpu_memory_mb,

            "optimization_plan": plan,

        }
