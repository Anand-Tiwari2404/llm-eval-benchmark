from src.utils.dataset_loader import build_seed_dataset

if __name__ == "__main__":
    cases = build_seed_dataset(
        squad_samples=100,
        openbookqa_samples=100,
        output_path="data/seed_dataset.json"
    )
    print(f"\n✅ Done! {len(cases)} test cases saved to data/seed_dataset.json")

    # Preview first 3 cases
    print("\n🔍 Preview:")
    for case in cases[:3]:
        print(f"\n  [{case.source}] {case.category.value} | {case.difficulty.value}")
        print(f"  Q: {case.question[:80]}...")
        print(f"  A: {case.reference_answer}")