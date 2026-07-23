
import math


class AllocatorCorrectionEstimator:
    def __init__(
        self,
        default_padding_ratio=0.10,
        min_padding_mb=8.0,
        rounding_mb=2.0,
    ):
        self.default_padding_ratio = default_padding_ratio
        self.min_padding_mb = min_padding_mb
        self.rounding_mb = rounding_mb

    def round_up(self, value, multiple):
        if multiple <= 0:
            return value
        return math.ceil(value / multiple) * multiple

    def estimate_reserved_memory_mb(
        self,
        predicted_allocated_mb,
        padding_ratio=None,
        min_padding_mb=None,
        rounding_mb=None,
    ):
        if padding_ratio is None:
            padding_ratio = self.default_padding_ratio

        if min_padding_mb is None:
            min_padding_mb = self.min_padding_mb

        if rounding_mb is None:
            rounding_mb = self.rounding_mb

        padding_mb = predicted_allocated_mb * padding_ratio

        if padding_mb < min_padding_mb:
            padding_mb = min_padding_mb

        raw_reserved = predicted_allocated_mb + padding_mb
        rounded_reserved = self.round_up(raw_reserved, rounding_mb)

        return {
            "predicted_reserved_MB": rounded_reserved,
            "allocator_padding_MB": padding_mb,
            "padding_ratio": padding_ratio,
            "rounding_mb": rounding_mb,
        }

    def estimate_padding_ratio_from_row(
        self,
        actual_allocated_mb,
        actual_reserved_mb,
    ):
        if actual_allocated_mb <= 0:
            return 0.0

        return (actual_reserved_mb - actual_allocated_mb) / actual_allocated_mb
