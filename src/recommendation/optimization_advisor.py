from dataclasses import dataclass
from typing import List

from src.estimators.training_memory_estimator import (
    TrainingMemoryEstimator,
)

from src.estimators.quantization_memory_estimator import (
    QuantizationMemoryEstimator,
)

from src.estimators.sparsity_memory_estimator import (
    SparsityMemoryEstimator,
)

from src.estimators.model_parallel_memory_estimator import (
    ModelParallelMemoryEstimator,
)


##############################################################################
# Dataclasses
##############################################################################

@dataclass
class QuantizationResult:
    dtype: str
    parameter_memory_mb: float
    reduction_percent: float


@dataclass
class SparsityResult:
    sparsity_percent: float
    parameter_memory_mb: float
    reduction_percent: float


@dataclass
class ModelParallelResult:
    num_devices: int
    per_device_memory_mb: float
    reduction_percent: float


##############################################################################
# Optimization Advisor
##############################################################################

class OptimizationAdvisor:
    """
    Parameter-memory optimization advisor.

    NOTE:
    This advisor estimates PARAMETER MEMORY ONLY.

    Runtime CUDA memory is handled separately by RuntimeAdvisor.
    """

    def __init__(self):

        self.training_estimator = TrainingMemoryEstimator()

        self.quantization_estimator = QuantizationMemoryEstimator()

        self.sparsity_estimator = SparsityMemoryEstimator()

        self.model_parallel_estimator = ModelParallelMemoryEstimator()

    ##########################################################################
    # Helper
    ##########################################################################

    def get_num_parameters(self, model_name):

        return self.training_estimator.get_num_parameters(
            model_name
        )

    ##########################################################################
    # Quantization Analysis
    ##########################################################################

    def analyze_quantization(
        self,
        model_name,
        dtypes=None,
    ) -> List[QuantizationResult]:

        if dtypes is None:
            dtypes = [
                "fp32",
                "fp16",
                "int8",
                "int4",
            ]

        num_parameters = self.get_num_parameters(
            model_name
        )

        results = []

        for dtype in dtypes:

            estimate = self.quantization_estimator.estimate(
                model_name=model_name,
                num_parameters=num_parameters,
                dtype=dtype,
            )

            results.append(
                QuantizationResult(
                    dtype=dtype,
                    parameter_memory_mb=estimate[
                        "effective_parameter_memory_MB"
                    ],
                    reduction_percent=estimate[
                        "effective_reduction_percent"
                    ],
                )
            )

        return results

    ##########################################################################
    # Convenience Function
    ##########################################################################

    def print_quantization_table(
        self,
        model_name,
    ):

        results = self.analyze_quantization(
            model_name
        )

        print("=" * 65)
        print(f"Quantization Analysis : {model_name}")
        print("=" * 65)

        print(
            f"{'DType':<10}"
            f"{'Memory (MB)':>15}"
            f"{'Reduction %':>18}"
        )

        print("-" * 65)

        for r in results:

            print(
                f"{r.dtype:<10}"
                f"{r.parameter_memory_mb:>15.2f}"
                f"{r.reduction_percent:>18.2f}"
            )

        print("=" * 65)

    ##########################################################################
    # Sparsity Analysis
    ##########################################################################

    def analyze_sparsity(
        self,
        model_name,
        sparsity_levels=None,
    ) -> List[SparsityResult]:
        """
        Analyze parameter memory under different sparsity levels.

        Returns
        -------
        List[SparsityResult]
        """

        if sparsity_levels is None:
            sparsity_levels = [
                0.25,
                0.50,
                0.75,
                0.90,
            ]

        num_parameters = self.get_num_parameters(
            model_name
        )

        results = []

        for sparsity in sparsity_levels:

            estimate = self.sparsity_estimator.estimate(
                model_name=model_name,
                num_parameters=num_parameters,
                sparsity=sparsity,
            )

            results.append(

                SparsityResult(

                    sparsity_percent=
                    estimate["sparsity_percent"],

                    parameter_memory_mb=
                    estimate[
                        "sparse_total_parameter_memory_MB"
                    ],

                    reduction_percent=
                    estimate[
                        "sparse_reduction_percent"
                    ],
                )

            )

        return results

    ##########################################################################
    # Pretty Print Sparsity Table
    ##########################################################################

    def print_sparsity_table(
        self,
        model_name,
    ):

        results = self.analyze_sparsity(
            model_name
        )

        print("=" * 70)
        print(f"Sparsity Analysis : {model_name}")
        print("=" * 70)

        print(
            f"{'Sparsity':<15}"
            f"{'Memory (MB)':>18}"
            f"{'Reduction %':>18}"
        )

        print("-" * 70)

        for r in results:

            print(
                f"{str(int(r.sparsity_percent))+'%':<15}"
                f"{r.parameter_memory_mb:>18.2f}"
                f"{r.reduction_percent:>18.2f}"
            )

        print("=" * 70)

    ##########################################################################
    # Model Parallel Analysis
    ##########################################################################

    def analyze_model_parallel(
        self,
        model_name,
        dtype="fp32",
        device_counts=None,
    ) -> List[ModelParallelResult]:

        if device_counts is None:
            device_counts = [1, 2, 4, 8]

        num_parameters = self.get_num_parameters(model_name)

        results = []

        for devices in device_counts:

            estimate = self.model_parallel_estimator.estimate(
                model_name=model_name,
                num_parameters=num_parameters,
                dtype=dtype,
                num_devices=devices,
            )

            results.append(

                ModelParallelResult(

                    num_devices=devices,

                    per_device_memory_mb=
                    estimate["estimated_per_device_memory_MB"],

                    reduction_percent=
                    estimate["effective_reduction_percent"],
                )

            )

        return results


    ##########################################################################
    # Pretty Print Model Parallel Table
    ##########################################################################

    def print_model_parallel_table(
        self,
        model_name,
        dtype="fp32",
    ):

        results = self.analyze_model_parallel(
            model_name=model_name,
            dtype=dtype,
        )

        print("=" * 70)
        print(f"Model Parallel Analysis : {model_name}")
        print("=" * 70)

        print(
            f"{'Devices':<15}"
            f"{'Memory / GPU (MB)':>22}"
            f"{'Reduction %':>18}"
        )

        print("-" * 70)

        for r in results:

            print(
                f"{r.num_devices:<15}"
                f"{r.per_device_memory_mb:>22.2f}"
                f"{r.reduction_percent:>18.2f}"
            )

        print("=" * 70)
