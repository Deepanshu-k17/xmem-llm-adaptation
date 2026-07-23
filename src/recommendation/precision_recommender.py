"""
Precision Recommender
"""


class PrecisionRecommender:

    @staticmethod
    def recommend(
        predicted_memory_mb,
        gpu_memory_mb,
    ):

        fp16 = predicted_memory_mb * 0.5
        bf16 = predicted_memory_mb * 0.5

        fp16_fit = fp16 <= gpu_memory_mb
        bf16_fit = bf16 <= gpu_memory_mb

        if fp16_fit:

            recommendation = (
                "Switching to FP16 should fit within GPU memory."
            )

        elif bf16_fit:

            recommendation = (
                "Switching to BF16 should fit within GPU memory."
            )

        else:

            recommendation = (
                "Mixed precision alone is insufficient. "
                "Reduce batch size or enable gradient checkpointing."
            )

        return {

            "fp16_memory_mb": fp16,
            "bf16_memory_mb": bf16,

            "fp16_fits": fp16_fit,
            "bf16_fits": bf16_fit,

            "recommendation": recommendation,
        }
