
import math


class QuantizationMemoryEstimator:
    """
    Analytical quantization memory estimator.

    This estimator predicts parameter memory under fp32, fp16, int8, and int4
    storage formats.

    Important:
    - This is analytical estimation, not real quantized model execution.
    - int8/int4 estimates include optional grouped metadata overhead.
    - It estimates parameter memory, not full runtime CUDA memory.
    """

    DTYPE_CONFIG = {
        "fp32": {
            "bits_per_parameter": 32,
            "bytes_per_parameter": 4.0,
            "is_quantized": False,
        },
        "fp16": {
            "bits_per_parameter": 16,
            "bytes_per_parameter": 2.0,
            "is_quantized": False,
        },
        "int8": {
            "bits_per_parameter": 8,
            "bytes_per_parameter": 1.0,
            "is_quantized": True,
        },
        "int4": {
            "bits_per_parameter": 4,
            "bytes_per_parameter": 0.5,
            "is_quantized": True,
        },
    }

    METADATA_CONFIG = {
        "none": {
            "scale_bytes": 0.0,
            "zero_point_bytes": 0.0,
        },
        "scale_only_fp16": {
            "scale_bytes": 2.0,
            "zero_point_bytes": 0.0,
        },
        "scale_plus_zero_point": {
            "scale_bytes": 2.0,
            "zero_point_bytes": 2.0,
        },
    }

    def __init__(
        self,
        default_group_size=128,
        default_metadata_case="scale_plus_zero_point",
    ):
        if default_group_size <= 0:
            raise ValueError("default_group_size must be positive.")

        if default_metadata_case not in self.METADATA_CONFIG:
            raise ValueError(
                f"Unsupported metadata case: {default_metadata_case}. "
                f"Supported: {list(self.METADATA_CONFIG.keys())}"
            )

        self.default_group_size = default_group_size
        self.default_metadata_case = default_metadata_case

    def supported_dtypes(self):
        return list(self.DTYPE_CONFIG.keys())

    def supported_metadata_cases(self):
        return list(self.METADATA_CONFIG.keys())

    def get_dtype_config(self, dtype):
        dtype = dtype.lower()

        if dtype not in self.DTYPE_CONFIG:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Supported: {self.supported_dtypes()}"
            )

        return self.DTYPE_CONFIG[dtype]

    def get_metadata_config(self, metadata_case):
        if metadata_case not in self.METADATA_CONFIG:
            raise ValueError(
                f"Unsupported metadata case: {metadata_case}. "
                f"Supported: {self.supported_metadata_cases()}"
            )

        return self.METADATA_CONFIG[metadata_case]

    def estimate_raw_parameter_memory_mb(self, num_parameters, dtype):
        """
        Estimate raw parameter memory without metadata overhead.
        """
        if num_parameters < 0:
            raise ValueError("num_parameters must be non-negative.")

        dtype_config = self.get_dtype_config(dtype)
        bytes_per_parameter = dtype_config["bytes_per_parameter"]

        return num_parameters * bytes_per_parameter / (1024 ** 2)

    def estimate_metadata_memory_mb(
        self,
        num_parameters,
        dtype,
        group_size=None,
        metadata_case=None,
    ):
        """
        Estimate grouped quantization metadata memory.

        Metadata is added only for int8/int4 by default.
        fp32/fp16 return 0.
        """
        if num_parameters < 0:
            raise ValueError("num_parameters must be non-negative.")

        dtype = dtype.lower()
        dtype_config = self.get_dtype_config(dtype)

        if not dtype_config["is_quantized"]:
            return 0.0

        if group_size is None:
            group_size = self.default_group_size

        if metadata_case is None:
            metadata_case = self.default_metadata_case

        if group_size <= 0:
            raise ValueError("group_size must be positive.")

        metadata_config = self.get_metadata_config(metadata_case)

        metadata_bytes_per_group = (
            metadata_config["scale_bytes"] + metadata_config["zero_point_bytes"]
        )

        num_groups = math.ceil(num_parameters / group_size)

        return num_groups * metadata_bytes_per_group / (1024 ** 2)

    def estimate_effective_parameter_memory_mb(
        self,
        num_parameters,
        dtype,
        group_size=None,
        metadata_case=None,
    ):
        """
        Estimate parameter memory including quantization metadata overhead.
        """
        raw_memory = self.estimate_raw_parameter_memory_mb(num_parameters, dtype)
        metadata_memory = self.estimate_metadata_memory_mb(
            num_parameters=num_parameters,
            dtype=dtype,
            group_size=group_size,
            metadata_case=metadata_case,
        )

        return raw_memory + metadata_memory

    def estimate_reduction_vs_fp32_percent(
        self,
        num_parameters,
        dtype,
        group_size=None,
        metadata_case=None,
    ):
        """
        Estimate effective parameter-memory reduction compared with fp32.
        """
        fp32_memory = self.estimate_effective_parameter_memory_mb(
            num_parameters=num_parameters,
            dtype="fp32",
            group_size=group_size,
            metadata_case=metadata_case,
        )

        target_memory = self.estimate_effective_parameter_memory_mb(
            num_parameters=num_parameters,
            dtype=dtype,
            group_size=group_size,
            metadata_case=metadata_case,
        )

        if fp32_memory == 0:
            return 0.0

        return (fp32_memory - target_memory) / fp32_memory * 100

    def estimate(
        self,
        model_name,
        num_parameters,
        dtype,
        group_size=None,
        metadata_case=None,
    ):
        """
        Return a complete estimate dictionary for one model and dtype.
        """
        dtype = dtype.lower()

        if group_size is None:
            group_size = self.default_group_size

        if metadata_case is None:
            metadata_case = self.default_metadata_case

        dtype_config = self.get_dtype_config(dtype)

        raw_parameter_memory_mb = self.estimate_raw_parameter_memory_mb(
            num_parameters=num_parameters,
            dtype=dtype,
        )

        metadata_memory_mb = self.estimate_metadata_memory_mb(
            num_parameters=num_parameters,
            dtype=dtype,
            group_size=group_size,
            metadata_case=metadata_case,
        )

        effective_parameter_memory_mb = raw_parameter_memory_mb + metadata_memory_mb

        fp32_parameter_memory_mb = self.estimate_effective_parameter_memory_mb(
            num_parameters=num_parameters,
            dtype="fp32",
            group_size=group_size,
            metadata_case=metadata_case,
        )

        reduction_percent = self.estimate_reduction_vs_fp32_percent(
            num_parameters=num_parameters,
            dtype=dtype,
            group_size=group_size,
            metadata_case=metadata_case,
        )

        metadata_overhead_percent_of_raw = (
            metadata_memory_mb / raw_parameter_memory_mb * 100
            if raw_parameter_memory_mb > 0
            else 0.0
        )

        num_groups = (
            math.ceil(num_parameters / group_size)
            if dtype_config["is_quantized"]
            else 0
        )

        return {
            "model_name": model_name,
            "num_parameters": int(num_parameters),
            "dtype": dtype,
            "bits_per_parameter": dtype_config["bits_per_parameter"],
            "bytes_per_parameter": dtype_config["bytes_per_parameter"],
            "is_quantized": dtype_config["is_quantized"],
            "group_size": int(group_size),
            "metadata_case": metadata_case,
            "num_groups": int(num_groups),
            "raw_parameter_memory_MB": raw_parameter_memory_mb,
            "metadata_memory_MB": metadata_memory_mb,
            "effective_parameter_memory_MB": effective_parameter_memory_mb,
            "fp32_parameter_memory_MB": fp32_parameter_memory_mb,
            "effective_reduction_percent": reduction_percent,
            "metadata_overhead_percent_of_raw_quantized": metadata_overhead_percent_of_raw,
            "scope_note": "Analytical parameter-memory estimate, not real quantized runtime execution.",
        }

    def estimate_many(
        self,
        models,
        dtypes=None,
        group_sizes=None,
        metadata_case=None,
    ):
        """
        Estimate quantization memory for multiple models/dtypes/group sizes.

        models format:
        [
            {"model_name": "gpt2", "num_parameters": 124439808},
            ...
        ]
        """
        if dtypes is None:
            dtypes = self.supported_dtypes()

        if group_sizes is None:
            group_sizes = [self.default_group_size]

        rows = []

        for model in models:
            model_name = model["model_name"]
            num_parameters = model["num_parameters"]

            for dtype in dtypes:
                for group_size in group_sizes:
                    rows.append(
                        self.estimate(
                            model_name=model_name,
                            num_parameters=num_parameters,
                            dtype=dtype,
                            group_size=group_size,
                            metadata_case=metadata_case,
                        )
                    )

        return rows
