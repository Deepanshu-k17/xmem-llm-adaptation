
class SparsityMemoryEstimator:
    """
    Analytical sparsity memory estimator.

    This estimator predicts parameter-memory storage under dense fp32
    and simple unstructured sparse fp32 assumptions.

    Important:
    - This is analytical storage estimation, not real sparse runtime profiling.
    - It does not run sparse CUDA kernels.
    - It estimates parameter memory only, not full runtime CUDA memory.
    """

    STORAGE_CONFIG = {
        "dense_fp32": {
            "value_bytes": 4.0,
            "index_bytes_per_nonzero": 0.0,
            "uses_index_metadata": False,
        },
        "unstructured_sparse_fp32": {
            "value_bytes": 4.0,
            "index_bytes_per_nonzero": 4.0,
            "uses_index_metadata": True,
        },
    }

    def __init__(self, default_storage_type="unstructured_sparse_fp32"):
        if default_storage_type not in self.STORAGE_CONFIG:
            raise ValueError(
                f"Unsupported storage type: {default_storage_type}. "
                f"Supported: {list(self.STORAGE_CONFIG.keys())}"
            )

        self.default_storage_type = default_storage_type

    def supported_storage_types(self):
        return list(self.STORAGE_CONFIG.keys())

    def get_storage_config(self, storage_type):
        if storage_type not in self.STORAGE_CONFIG:
            raise ValueError(
                f"Unsupported storage type: {storage_type}. "
                f"Supported: {self.supported_storage_types()}"
            )

        return self.STORAGE_CONFIG[storage_type]

    def estimate_dense_parameter_memory_mb(self, num_parameters, value_bytes=4.0):
        """
        Estimate dense parameter memory.
        """
        if num_parameters < 0:
            raise ValueError("num_parameters must be non-negative.")

        return num_parameters * value_bytes / (1024 ** 2)

    def estimate_sparse_parameter_memory_mb(
        self,
        num_parameters,
        sparsity,
        storage_type=None,
    ):
        """
        Estimate sparse parameter memory.

        sparsity should be between 0 and 1.
        Example:
        0.75 means 75% of weights are zero.
        """
        if num_parameters < 0:
            raise ValueError("num_parameters must be non-negative.")

        if sparsity < 0 or sparsity > 1:
            raise ValueError("sparsity must be between 0 and 1.")

        if storage_type is None:
            storage_type = self.default_storage_type

        config = self.get_storage_config(storage_type)

        nonzero_fraction = 1.0 - sparsity
        nonzero_parameters = num_parameters * nonzero_fraction

        value_memory_mb = (
            nonzero_parameters * config["value_bytes"] / (1024 ** 2)
        )

        index_memory_mb = (
            nonzero_parameters
            * config["index_bytes_per_nonzero"]
            / (1024 ** 2)
        )

        total_sparse_memory_mb = value_memory_mb + index_memory_mb

        return {
            "nonzero_fraction": nonzero_fraction,
            "nonzero_parameters": int(nonzero_parameters),
            "sparse_value_memory_MB": value_memory_mb,
            "sparse_index_memory_MB": index_memory_mb,
            "sparse_total_parameter_memory_MB": total_sparse_memory_mb,
        }

    def estimate_reduction_vs_dense_percent(
        self,
        num_parameters,
        sparsity,
        storage_type=None,
        dense_value_bytes=4.0,
    ):
        """
        Estimate sparse parameter-memory reduction compared with dense fp32.
        """
        dense_memory = self.estimate_dense_parameter_memory_mb(
            num_parameters=num_parameters,
            value_bytes=dense_value_bytes,
        )

        sparse_result = self.estimate_sparse_parameter_memory_mb(
            num_parameters=num_parameters,
            sparsity=sparsity,
            storage_type=storage_type,
        )

        sparse_memory = sparse_result["sparse_total_parameter_memory_MB"]

        if dense_memory == 0:
            return 0.0

        return (dense_memory - sparse_memory) / dense_memory * 100

    def estimate(
        self,
        model_name,
        num_parameters,
        sparsity,
        storage_type=None,
    ):
        """
        Return full sparsity estimate for one model and sparsity level.
        """
        if storage_type is None:
            storage_type = self.default_storage_type

        config = self.get_storage_config(storage_type)

        dense_memory = self.estimate_dense_parameter_memory_mb(
            num_parameters=num_parameters,
            value_bytes=4.0,
        )

        sparse_result = self.estimate_sparse_parameter_memory_mb(
            num_parameters=num_parameters,
            sparsity=sparsity,
            storage_type=storage_type,
        )

        reduction_percent = self.estimate_reduction_vs_dense_percent(
            num_parameters=num_parameters,
            sparsity=sparsity,
            storage_type=storage_type,
        )

        sparse_memory = sparse_result["sparse_total_parameter_memory_MB"]

        overhead_vs_dense_percent = (
            (sparse_memory - dense_memory) / dense_memory * 100
            if dense_memory > 0
            else 0.0
        )

        return {
            "model_name": model_name,
            "num_parameters": int(num_parameters),
            "storage_type": storage_type,
            "sparsity_percent": sparsity * 100,
            "nonzero_fraction": sparse_result["nonzero_fraction"],
            "nonzero_parameters": sparse_result["nonzero_parameters"],
            "value_bytes": config["value_bytes"],
            "index_bytes_per_nonzero": config["index_bytes_per_nonzero"],
            "uses_index_metadata": config["uses_index_metadata"],
            "dense_fp32_parameter_memory_MB": dense_memory,
            "sparse_value_memory_MB": sparse_result["sparse_value_memory_MB"],
            "sparse_index_memory_MB": sparse_result["sparse_index_memory_MB"],
            "sparse_total_parameter_memory_MB": sparse_result["sparse_total_parameter_memory_MB"],
            "sparse_reduction_percent": reduction_percent,
            "sparse_overhead_vs_dense_percent": overhead_vs_dense_percent,
            "scope_note": "Analytical sparse parameter-memory estimate, not real sparse runtime execution.",
        }

    def estimate_many(
        self,
        models,
        sparsity_levels,
        storage_type=None,
    ):
        """
        Estimate sparse memory for many models and sparsity levels.

        models format:
        [
            {"model_name": "gpt2", "num_parameters": 124439808},
            ...
        ]
        """
        rows = []

        for model in models:
            for sparsity in sparsity_levels:
                rows.append(
                    self.estimate(
                        model_name=model["model_name"],
                        num_parameters=model["num_parameters"],
                        sparsity=sparsity,
                        storage_type=storage_type,
                    )
                )

        return rows
