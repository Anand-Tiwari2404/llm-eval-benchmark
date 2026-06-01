import json
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file, save_dataset
from src.evals.adversarial_generator import AdversarialGenerator

if __name__ == "__main__":
    config = load_config()

    # Load seed dataset
    print("📂 Loading seed dataset...")
    seed_cases = load_dataset_from_file("data/seed_dataset.json")

    # Generate adversarial cases
    generator = AdversarialGenerator(api_key=config.groq_api_key)
    adversarial_cases = generator.generate(
        seed_cases=seed_cases,
        num_adversarial=30
    )

    # Save adversarial cases separately
    save_dataset(adversarial_cases, "data/adversarial_dataset.json")

    # Preview 3 examples
    print("\n🔍 Preview of adversarial cases:")
    for case in adversarial_cases[:3]:
        print(f"\n  Strategy: {case.source}")
        print(f"  Original: {case.context[:80]}...")
        print(f"  Adversarial: {case.question[:80]}...")
        print(f"  Correct response: {case.reference_answer[:80]}...")

    print(f"\n🎯 Done! Adversarial dataset saved to data/adversarial_dataset.json")