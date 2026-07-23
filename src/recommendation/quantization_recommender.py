"""
Quantization Recommender
"""


class QuantizationRecommender:

    @staticmethod
    def recommend(
        predicted_memory_mb,
        gpu_memory_mb,
    ):

        int8 = predicted_memory_mb * 0.5
        int4 = predicted_memory_mb * 0.25

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
                "Even INT4 quantization is insufficient. "
                "Consider reducing the batch size or using model parallelism."
            )

            precision = "None"

        return {

            "recommended_precision": precision,

            "recommendation": recommendation,

            "int8_memory_mb": int8,

            "int4_memory_mb": int4,
        }
