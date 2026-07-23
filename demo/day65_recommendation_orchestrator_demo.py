
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.recommendation.recommendation_orchestrator import (
    RecommendationOrchestrator
)

engine = RecommendationOrchestrator()

tests = [

    (18000, 24000, 32),

    (22000, 24000, 64),

    (52000, 24000, 64),

]

for predicted, gpu, batch in tests:

    result = engine.recommend(
        predicted,
        gpu,
        batch,
    )

    print("=" * 70)

    print(
        f"Predicted Memory : {result['predicted_memory_mb']} MB"
    )

    print(
        f"GPU Memory       : {result['gpu_memory_mb']} MB"
    )

    print(
        f"Status           : {result['status']}"
    )

    print("\nOptimization Plan")

    for i, step in enumerate(
        result["optimization_plan"],
        start=1,
    ):

        print(f"{i}. {step}")

    print()
