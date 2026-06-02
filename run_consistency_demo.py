from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.evals.self_consistency import SelfConsistencyChecker

if __name__ == "__main__":
    config = load_config()

    # Load cases — mix of easy and hard
    all_cases = load_dataset_from_file("data/seed_dataset.json")

    # Take 5 cases for demo
    demo_cases = all_cases[:5]

    checker = SelfConsistencyChecker(api_key=config.groq_api_key)
    results = checker.check_batch(demo_cases, max_cases=5)

    # Detailed view
    print("\n🔍 Detailed results:")
    for result in results:
        status = "✅ CONSISTENT" if result.is_consistent else "⚠️  INCONSISTENT"
        print(f"\n  {status} | Score: {result.consistency_score}")
        print(f"  Q: {result.question[:70]}...")
        print(f"  Majority answer: {result.majority_answer[:80]}...")
        print(f"  Reasoning: {result.reasoning[:100]}...")
        print(f"  All samples:")
        for i, s in enumerate(result.samples):
            print(f"    [{i+1}] {s[:60]}...")