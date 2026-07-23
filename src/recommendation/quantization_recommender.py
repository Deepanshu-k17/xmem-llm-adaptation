"""
quantization_recommender.py
"""


class QuantizationRecommender:

    @staticmethod
    def estimate_int8(memory):
        return memory * 0.5

    @staticmethod
    def estimate_int4(memory):
        return memory * 0.25

    @staticmethod
    def recommend(predicted_memory_mb, gpu_memory_mb):

        int8 = QuantizationRecommender.estimate_int8(
            predicted_memory_mb
        )

        int4 = QuantizationRecommender.estimate_int4(
            predicted_memory_mb
        )

        int8_fit = int8 <= gpu_memory_mb
        int4_fit = int4 <= gpu_memory_mb

        if predicted_memory_mb <= gpu_memory_mb:

            recommendation = "No quantization is required."

            precision = "None"

        elif int8_fit:

            recommendation = (
                "INT8 quantization should fit within GPU memory."
            )

            precision = "INT8"

        elif int4_fit:

            recommendation = (
                "INT8 is insufficient. Use INT4 quantization."
            )

            precision = "INT4"

        else:

            recommendation = (
                "Even INT4 quantization is insufficient."
            )

            precision = "None"

        return {

            "current_memory_mb": predicted_memory_mb,

            "gpu_memory_mb": gpu_memory_mb,

            "int8_memory_mb": int8,

            "int4_memory_mb": int4,

            "int8_fits": int8_fit,

            "int4_fits": int4_fit,

            "recommended_precision": precision,

            "recommendation": recommendation,
        }
