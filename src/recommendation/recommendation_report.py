
from dataclasses import dataclass
from typing import List

from src.recommendation.runtime_advisor import RuntimeAdvisor
from src.recommendation.optimization_advisor import OptimizationAdvisor


##############################################################################
# Report Dataclass
##############################################################################

@dataclass
class RecommendationReport:

    model_name: str

    gpu_memory_mb: float

    task: str

    runtime_result: object

    quantization_results: List

    sparsity_results: List

    model_parallel_results: List


##############################################################################
# Recommendation Report Generator
##############################################################################

class RecommendationReportGenerator:

    def __init__(self):

        self.runtime_advisor = RuntimeAdvisor()

        self.optimization_advisor = OptimizationAdvisor()

    ##########################################################################

    def generate_report(
        self,
        model_name,
        gpu_memory_mb,
        task="inference",
        **kwargs,
    ):

        runtime = self.runtime_advisor.analyze(
            model_name=model_name,
            available_gpu_memory_mb=gpu_memory_mb,
            task=task,
            **kwargs,
        )

        quantization = self.optimization_advisor.analyze_quantization(
            model_name
        )

        sparsity = self.optimization_advisor.analyze_sparsity(
            model_name
        )

        model_parallel = self.optimization_advisor.analyze_model_parallel(
            model_name
        )

        return RecommendationReport(

            model_name=model_name,

            gpu_memory_mb=gpu_memory_mb,

            task=task,

            runtime_result=runtime,

            quantization_results=quantization,

            sparsity_results=sparsity,

            model_parallel_results=model_parallel,
        )

    ##########################################################################

    def print_report(
        self,
        model_name,
        gpu_memory_mb,
        task="inference",
        **kwargs,
    ):

        report = self.generate_report(

            model_name=model_name,

            gpu_memory_mb=gpu_memory_mb,

            task=task,

            **kwargs,
        )

        print("=" * 90)
        print("xMem Memory Recommendation Report")
        print("=" * 90)

        print()

        print(f"Model          : {report.model_name}")
        print(f"Task           : {report.task}")
        print(f"GPU Memory     : {report.gpu_memory_mb:.2f} MB")

        print()

        ####################################################################
        # Runtime
        ####################################################################

        runtime = report.runtime_result

        print("=" * 90)
        print("Runtime Analysis")
        print("=" * 90)

        print(f"Fits GPU              : {runtime.fits}")
        print(f"Estimated Reserved MB : {runtime.estimated_reserved_memory_mb:.2f}")
        print(f"Required + Margin MB  : {runtime.required_memory_with_margin_mb:.2f}")
        print(f"GPU Utilization (%)   : {runtime.utilization_percent:.2f}")
        print(runtime.message)

        ####################################################################
        # Quantization
        ####################################################################

        print()
        print("=" * 90)
        print("Quantization Analysis")
        print("=" * 90)

        print(
            f"{'DType':<10}"
            f"{'Memory(MB)':>18}"
            f"{'Reduction %':>18}"
        )

        print("-" * 90)

        for r in report.quantization_results:

            print(
                f"{r.dtype:<10}"
                f"{r.parameter_memory_mb:>18.2f}"
                f"{r.reduction_percent:>18.2f}"
            )

        ####################################################################
        # Sparsity
        ####################################################################

        print()
        print("=" * 90)
        print("Sparsity Analysis")
        print("=" * 90)

        print(
            f"{'Sparsity':<15}"
            f"{'Memory(MB)':>18}"
            f"{'Reduction %':>18}"
        )

        print("-" * 90)

        for r in report.sparsity_results:

            print(
                f"{int(r.sparsity_percent)}%".ljust(15)
                + f"{r.parameter_memory_mb:>18.2f}"
                + f"{r.reduction_percent:>18.2f}"
            )

        ####################################################################
        # Model Parallel
        ####################################################################

        print()
        print("=" * 90)
        print("Model Parallel Analysis")
        print("=" * 90)

        print(
            f"{'Devices':<15}"
            f"{'Memory/GPU(MB)':>20}"
            f"{'Reduction %':>18}"
        )

        print("-" * 90)

        for r in report.model_parallel_results:

            print(
                f"{r.num_devices:<15}"
                f"{r.per_device_memory_mb:>20.2f}"
                f"{r.reduction_percent:>18.2f}"
            )

        ####################################################################
        # Best Choices
        ####################################################################

        best_quant = min(
            report.quantization_results,
            key=lambda x: x.parameter_memory_mb,
        )

        best_sparse = min(
            report.sparsity_results,
            key=lambda x: x.parameter_memory_mb,
        )

        best_parallel = min(
            report.model_parallel_results,
            key=lambda x: x.per_device_memory_mb,
        )

        print()
        print("=" * 90)
        print("Best Optimization Choices")
        print("=" * 90)

        print(
            f"Best Quantization : "
            f"{best_quant.dtype} "
            f"({best_quant.parameter_memory_mb:.2f} MB)"
        )

        print(
            f"Best Sparsity     : "
            f"{best_sparse.sparsity_percent:.0f}% "
            f"({best_sparse.parameter_memory_mb:.2f} MB)"
        )

        print(
            f"Best Parallel     : "
            f"{best_parallel.num_devices} GPUs "
            f"({best_parallel.per_device_memory_mb:.2f} MB/GPU)"
        )

        print()

        print("=" * 90)

        if runtime.fits:

            print("FINAL RECOMMENDATION")

            print("✓ Current configuration already fits in GPU memory.")

            print("No optimization is strictly required.")

        else:

            print("FINAL RECOMMENDATION")

            print(
                f"Consider using {best_quant.dtype} precision "
                "or distributed execution."
            )

        print("=" * 90)
