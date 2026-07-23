"""
precision_recommender.py
"""


class PrecisionRecommender:

    @staticmethod
    def estimate_fp16(memory):
        return memory * 0.5

    @staticmethod
    def estimate_bf16(memory):
        return memory * 0.5

    @staticmethod
    def recommend(predicted_memory_mb, gpu_memory_mb):

        fp16 = PrecisionRecommender.estimate_fp16(
            predicted_memory_mb
        )

        bf16 = PrecisionRecommender.estimate_bf16(
            predicted_memory_mb
        )

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

            "current_memory_mb": predicted_memory_mb,

            "gpu_memory_mb": gpu_memory_mb,

            "fp16_memory_mb": fp16,

            "bf16_memory_mb": bf16,

            "fp16_fits": fp16_fit,

            "bf16_fits": bf16_fit,

            "recommendation": recommendation,
        }
