
class OptimizerStateEstimator:
    def __init__(self):
        self.optimizer_state_factors = {
            "sgd": 0.0,
            "sgd_momentum": 1.0,
            "adam": 2.0,
            "adamw": 2.0,
        }

        self.model_parameter_counts = {
            "sshleifer/tiny-gpt2": 102714,
            "tiny-gpt2": 102714,
            "distilgpt2": 81912576,
            "gpt2": 124439808,
        }

        self.dtype_bytes = {
            "fp32": 4,
            "float32": 4,
            "fp16": 2,
            "float16": 2,
            "bf16": 2,
            "bfloat16": 2,
        }

    def normalize_optimizer_name(self, optimizer_name):
        name = str(optimizer_name).lower().strip()

        if name in ["sgd", "torch.optim.sgd"]:
            return "sgd"

        if name in ["sgd_momentum", "momentum_sgd", "sgd_with_momentum"]:
            return "sgd_momentum"

        if name in ["adam", "torch.optim.adam"]:
            return "adam"

        if name in ["adamw", "torch.optim.adamw"]:
            return "adamw"

        return name

    def normalize_dtype(self, dtype):
        dtype = str(dtype).lower().strip()

        if dtype not in self.dtype_bytes:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. "
                f"Supported dtypes: {list(self.dtype_bytes.keys())}"
            )

        return dtype

    def get_num_parameters(self, model_name):
        model_name = str(model_name)

        if model_name not in self.model_parameter_counts:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {list(self.model_parameter_counts.keys())}"
            )

        return self.model_parameter_counts[model_name]

    def estimate_parameter_memory_mb(self, model_name, dtype="fp32"):
        dtype = self.normalize_dtype(dtype)

        num_parameters = self.get_num_parameters(model_name)
        bytes_per_param = self.dtype_bytes[dtype]

        memory_mb = (num_parameters * bytes_per_param) / (1024 ** 2)

        return memory_mb

    def get_optimizer_state_factor(self, optimizer_name):
        normalized_name = self.normalize_optimizer_name(optimizer_name)

        if normalized_name not in self.optimizer_state_factors:
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. "
                f"Supported optimizers: {list(self.optimizer_state_factors.keys())}"
            )

        return self.optimizer_state_factors[normalized_name]

    def estimate_optimizer_state_memory_mb(
        self,
        model_name,
        optimizer_name,
        dtype="fp32",
    ):
        normalized_name = self.normalize_optimizer_name(optimizer_name)

        parameter_memory_mb = self.estimate_parameter_memory_mb(
            model_name=model_name,
            dtype=dtype,
        )

        optimizer_state_factor = self.get_optimizer_state_factor(normalized_name)
        optimizer_state_memory_mb = parameter_memory_mb * optimizer_state_factor

        return {
            "model_name": model_name,
            "optimizer_name": normalized_name,
            "dtype": dtype,
            "num_parameters": self.get_num_parameters(model_name),
            "parameter_memory_MB": parameter_memory_mb,
            "optimizer_state_factor": optimizer_state_factor,
            "estimated_optimizer_state_MB": optimizer_state_memory_mb,
        }
