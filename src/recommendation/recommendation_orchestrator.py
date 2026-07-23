"""
Recommendation Orchestrator
"""

from src.recommendation.decision_engine import DecisionEngine
from src.recommendation.batch_size_recommender import BatchSizeRecommender
from src.recommendation.precision_recommender import PrecisionRecommender
from src.recommendation.quantization_recommender import QuantizationRecommender


class RecommendationOrchestrator:

    def __init__(self):

        self.decision = DecisionEngine()

    def recommend(
        self,
        predicted_memory_mb,
        gpu_memory_mb,
        current_batch_size,
    ):

        decision = self.decision.evaluate(
            predicted_memory_mb,
            gpu_memory_mb,
            current_batch_size,
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

            precision = PrecisionRecommender.recommend(
                predicted_memory_mb,
                gpu_memory_mb,
            )

            plan.append(
                precision["recommendation"]
            )

        else:

            plan.append(
                "Predicted memory exceeds GPU memory."
            )

            precision = PrecisionRecommender.recommend(
                predicted_memory_mb,
                gpu_memory_mb,
            )

            quant = QuantizationRecommender.recommend(
                predicted_memory_mb,
                gpu_memory_mb,
            )

            batch = BatchSizeRecommender.recommend(
                current_batch_size,
                predicted_memory_mb,
                gpu_memory_mb,
            )

            plan.append(
                precision["recommendation"]
            )

            plan.append(
                quant["recommendation"]
            )

            plan.append(
                f"Reduce batch size from "
                f"{batch['current_batch_size']} "
                f"to "
                f"{batch['recommended_batch_size']}."
            )

        return {

            "predicted_memory_mb": predicted_memory_mb,

            "gpu_memory_mb": gpu_memory_mb,

            "status": decision["status"],

            "optimization_plan": plan,
        }
