"""
batch_size_recommender.py

Recommend a batch size based on predicted memory.
"""


class BatchSizeRecommender:

    @staticmethod
    def recommend(
        current_batch_size,
        predicted_memory_mb,
        gpu_memory_mb,
    ):

        if current_batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if predicted_memory_mb <= 0:
            raise ValueError("Predicted memory must be positive.")

        if gpu_memory_mb <= 0:
            raise ValueError("GPU memory must be positive.")

        scaling_factor = gpu_memory_mb / predicted_memory_mb

        recommended_batch = max(
            1,
            int(current_batch_size * scaling_factor)
        )

        return {
            "current_batch_size": current_batch_size,
            "recommended_batch_size": recommended_batch,
            "scaling_factor": scaling_factor,
        }
