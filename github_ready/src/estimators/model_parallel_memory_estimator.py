
class ModelParallelMemoryEstimator:
    """
    Analytical model-parallel memory estimator.

    This estimator predicts per-device parameter memory when model
    parameters are partitioned across multiple devices.

    Important:
    - This is analytical memory estimation, not real multi-GPU profiling.
    - It does not run tensor parallelism, pipeline parallelism, NCCL, or distributed PyTorch.
    - It estimates parameter-memory partitioning only, not full runtime memory.
    """

    DTYPE_BYTES = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
    }

    def __init__(
        self,
        replication_overhead_percent=0.05,
        communication_buffer_percent=0.03,
    ):
        if replication_overhead_percent < 0:
            raise ValueError("replication_overhead_percent must be non-negative.")

        if communication_buffer_percent < 0:
            raise ValueError("communication_buffer_percent must be non-negative.")

        self.replication_overhead_percent = replication_overhead_percent
        self.communication_buffer_percent = communication_buffer_percent

    def supported_dtypes(self):
        return list(self.DTYPE_BYTES.keys())

    def estimate_parameter_memory_mb(self, num_parameters, dtype="fp32"):
        """
        Estimate total dense parameter memory for a dtype.
        """
        if num_parameters < 0:
            raise ValueError("num_parameters must be non-negative.")

        if dtype not in self.DTYPE_BYTES:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported: {self.supported_dtypes()}"
            )

        return num_parameters * self.DTYPE_BYTES[dtype] / (1024 ** 2)

    def estimate_ideal_partitioned_memory_mb(
        self,
        total_parameter_memory_mb,
        num_devices,
    ):
        """
        Estimate ideal per-device memory with perfect parameter partitioning.
        """
        if num_devices <= 0:
            raise ValueError("num_devices must be positive.")

        return total_parameter_memory_mb / num_devices

    def estimate_overhead_memory_mb(self, total_parameter_memory_mb):
        """
        Estimate replicated overhead and communication buffer memory.
        """
        replicated_overhead_memory_mb = (
            total_parameter_memory_mb * self.replication_overhead_percent
        )

        communication_buffer_memory_mb = (
            total_parameter_memory_mb * self.communication_buffer_percent
        )

        total_overhead_memory_mb = (
            replicated_overhead_memory_mb + communication_buffer_memory_mb
        )

        return {
            "replicated_overhead_memory_MB": replicated_overhead_memory_mb,
            "communication_buffer_memory_MB": communication_buffer_memory_mb,
            "total_overhead_memory_MB": total_overhead_memory_mb,
        }

    def estimate(
        self,
        model_name,
        num_parameters,
        dtype="fp32",
        num_devices=1,
    ):
        """
        Return full per-device model-parallel memory estimate.
        """
        if num_devices <= 0:
            raise ValueError("num_devices must be positive.")

        total_parameter_memory_mb = self.estimate_parameter_memory_mb(
            num_parameters=num_parameters,
            dtype=dtype,
        )

        ideal_partitioned_memory_mb = self.estimate_ideal_partitioned_memory_mb(
            total_parameter_memory_mb=total_parameter_memory_mb,
            num_devices=num_devices,
        )

        overhead = self.estimate_overhead_memory_mb(
            total_parameter_memory_mb=total_parameter_memory_mb
        )

        estimated_per_device_memory_mb = (
            ideal_partitioned_memory_mb
            + overhead["total_overhead_memory_MB"]
        )

        ideal_reduction_percent = (
            (total_parameter_memory_mb - ideal_partitioned_memory_mb)
            / total_parameter_memory_mb
            * 100
            if total_parameter_memory_mb > 0
            else 0.0
        )

        effective_reduction_percent = (
            (total_parameter_memory_mb - estimated_per_device_memory_mb)
            / total_parameter_memory_mb
            * 100
            if total_parameter_memory_mb > 0
            else 0.0
        )

        overhead_fraction_of_total_percent = (
            overhead["total_overhead_memory_MB"]
            / total_parameter_memory_mb
            * 100
            if total_parameter_memory_mb > 0
            else 0.0
        )

        return {
            "model_name": model_name,
            "num_parameters": int(num_parameters),
            "dtype": dtype,
            "num_devices": int(num_devices),
            "total_parameter_memory_MB": total_parameter_memory_mb,
            "ideal_partitioned_parameter_memory_MB": ideal_partitioned_memory_mb,
            "replication_overhead_percent": self.replication_overhead_percent * 100,
            "communication_buffer_percent": self.communication_buffer_percent * 100,
            "replicated_overhead_memory_MB": overhead["replicated_overhead_memory_MB"],
            "communication_buffer_memory_MB": overhead["communication_buffer_memory_MB"],
            "total_overhead_memory_MB": overhead["total_overhead_memory_MB"],
            "estimated_per_device_memory_MB": estimated_per_device_memory_mb,
            "ideal_reduction_percent": ideal_reduction_percent,
            "effective_reduction_percent": effective_reduction_percent,
            "overhead_fraction_of_total_percent": overhead_fraction_of_total_percent,
            "scope_note": "Analytical model-parallel memory estimate, not real multi-GPU execution.",
        }

    def estimate_many(
        self,
        models,
        dtypes,
        device_counts,
    ):
        """
        Estimate model-parallel memory for many models, dtypes, and device counts.

        models format:
        [
            {"model_name": "gpt2", "num_parameters": 124439808},
            ...
        ]
        """
        rows = []

        for model in models:
            for dtype in dtypes:
                for num_devices in device_counts:
                    rows.append(
                        self.estimate(
                            model_name=model["model_name"],
                            num_parameters=model["num_parameters"],
                            dtype=dtype,
                            num_devices=num_devices,
                        )
                    )

        return rows
