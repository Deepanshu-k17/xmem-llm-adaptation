
from estimators.precision_estimator import PrecisionAwareEstimator
from estimators.quantization_memory_estimator import QuantizationMemoryEstimator
from estimators.sparsity_memory_estimator import SparsityMemoryEstimator
from estimators.model_parallel_memory_estimator import ModelParallelMemoryEstimator


class EstimatorRegistry:

    def __init__(self):

        self.registry = {
            "precision": PrecisionAwareEstimator,
            "quantization": QuantizationMemoryEstimator,
            "sparsity": SparsityMemoryEstimator,
            "model_parallel": ModelParallelMemoryEstimator,
        }

    def available_estimators(self):
        return list(self.registry.keys())

    def create(self, estimator_name):

        if estimator_name not in self.registry:
            raise ValueError(f"Unknown estimator: {estimator_name}")

        return self.registry[estimator_name]()
