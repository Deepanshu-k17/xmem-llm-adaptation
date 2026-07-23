import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommendation.recommendation_orchestrator import RecommendationOrchestrator


def main():

    parser = argparse.ArgumentParser(description="xMem GPU Memory Advisor")

    parser.add_argument("--memory", type=float, required=True)
    parser.add_argument("--gpu", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)

    args = parser.parse_args()

    engine = RecommendationOrchestrator()

    result = engine.recommend(
        predicted_memory_mb=args.memory,
        gpu_memory_mb=args.gpu,
        current_batch_size=args.batch_size,
    )

    print("=" * 65)
    print("xMem GPU Memory Advisor")
    print("=" * 65)

    print(f"Predicted Memory : {result['predicted_memory_mb']:.2f} MB")
    print(f"GPU Memory       : {result['gpu_memory_mb']:.2f} MB")
    print(f"Status           : {result['status']}")

    print("\nOptimization Plan")

    for i, step in enumerate(result["optimization_plan"], start=1):
        print(f"{i}. {step}")

    print("=" * 65)


if __name__ == "__main__":
    main()
