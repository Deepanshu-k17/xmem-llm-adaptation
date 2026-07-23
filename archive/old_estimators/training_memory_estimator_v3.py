
from estimators.optimizer_state_estimator import OptimizerStateEstimator


class TrainingMemoryEstimatorV3:
    def __init__(
        self,
        activation_factor=0.004,
        adamw_backward_temp_factor=0.65,
        adam_backward_temp_factor=0.65,
        sgd_backward_temp_factor=0.35,
        sgd_momentum_backward_temp_factor=0.45,
        default_backward_temp_factor=0.50,
        framework_overhead_mb=20.0,
        allocator_padding_ratio=0.08,
        allocator_min_padding_mb=32.0,
    ):
        self.optimizer_estimator = OptimizerStateEstimator()

        self.activation_factor = activation_factor

        self.backward_temp_factors = {
            "adamw": adamw_backward_temp_factor,
            "adam": adam_backward_temp_factor,
            "sgd": sgd_backward_temp_factor,
            "sgd_momentum": sgd_momentum_backward_temp_factor,
        }

        self.default_backward_temp_factor = default_backward_temp_factor
        self.framework_overhead_mb = framework_overhead_mb
        self.allocator_padding_ratio = allocator_padding_ratio
        self.allocator_min_padding_mb = allocator_min_padding_mb

    def get_backward_temp_factor(self, optimizer_name):
        opt = optimizer_name.lower()
        return self.backward_temp_factors.get(opt, self.default_backward_temp_factor)

    def estimate_gradient_memory_mb(self, model_name, dtype="fp32"):
        return self.optimizer_estimator.estimate_parameter_memory_mb(
            model_name=model_name,
            dtype=dtype,
        )

    def estimate_activation_memory_mb(
        self,
        model_name,
        batch_size,
        input_tokens,
        dtype="fp32",
    ):
        parameter_memory_mb = self.optimizer_estimator.estimate_parameter_memory_mb(
            model_name=model_name,
            dtype=dtype,
        )

        activation_memory_mb = (
            parameter_memory_mb
            * self.activation_factor
            * batch_size
            * input_tokens
        )

        return activation_memory_mb

    def estimate_backward_temp_memory_mb(
        self,
        model_name,
        optimizer_name,
        dtype="fp32",
    ):
        parameter_memory_mb = self.optimizer_estimator.estimate_parameter_memory_mb(
            model_name=model_name,
            dtype=dtype,
        )

        backward_temp_factor = self.get_backward_temp_factor(optimizer_name)

        return parameter_memory_mb * backward_temp_factor

    def estimate_allocator_padding_mb(self, predicted_allocated_mb):
        ratio_padding = predicted_allocated_mb * self.allocator_padding_ratio
        return max(ratio_padding, self.allocator_min_padding_mb)

    def estimate_training_memory_mb(
        self,
        model_name,
        batch_size,
        input_tokens,
        optimizer_name,
        dtype="fp32",
    ):
        optimizer_result = self.optimizer_estimator.estimate_optimizer_state_memory_mb(
            model_name=model_name,
            optimizer_name=optimizer_name,
            dtype=dtype,
        )

        parameter_memory_mb = optimizer_result["parameter_memory_MB"]

        gradient_memory_mb = self.estimate_gradient_memory_mb(
            model_name=model_name,
            dtype=dtype,
        )

        optimizer_state_memory_mb = optimizer_result["estimated_optimizer_state_MB"]

        activation_memory_mb = self.estimate_activation_memory_mb(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
            dtype=dtype,
        )

        backward_temp_factor = self.get_backward_temp_factor(optimizer_name)

        backward_temp_memory_mb = self.estimate_backward_temp_memory_mb(
            model_name=model_name,
            optimizer_name=optimizer_name,
            dtype=dtype,
        )

        predicted_peak_allocated_mb = (
            parameter_memory_mb
            + gradient_memory_mb
            + optimizer_state_memory_mb
            + activation_memory_mb
            + backward_temp_memory_mb
            + self.framework_overhead_mb
        )

        allocator_padding_mb = self.estimate_allocator_padding_mb(
            predicted_peak_allocated_mb
        )

        predicted_peak_reserved_mb = predicted_peak_allocated_mb + allocator_padding_mb

        return {
            "model_name": model_name,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "optimizer_name": optimizer_result["optimizer_name"],
            "dtype": dtype,

            "parameter_memory_MB": parameter_memory_mb,
            "gradient_memory_MB": gradient_memory_mb,
            "optimizer_state_memory_MB": optimizer_state_memory_mb,
            "activation_memory_MB": activation_memory_mb,
            "backward_temp_factor": backward_temp_factor,
            "backward_temp_memory_MB": backward_temp_memory_mb,
            "framework_overhead_MB": self.framework_overhead_mb,
            "allocator_padding_MB": allocator_padding_mb,
            "allocator_padding_ratio": self.allocator_padding_ratio,
            "allocator_min_padding_MB": self.allocator_min_padding_mb,

            "predicted_peak_allocated_MB": predicted_peak_allocated_mb,
            "predicted_peak_reserved_MB": predicted_peak_reserved_mb,
        }
