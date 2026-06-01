import pytest
import os
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.utils.data_models import Category, Difficulty
from src.evals.adversarial_generator import AdversarialGenerator


def test_adversarial_dataset_exists():
    assert os.path.exists("data/adversarial_dataset.json"), \
        "Run generate_adversarial.py first!"
    print("\n✅ adversarial_dataset.json exists")


def test_adversarial_cases_load_correctly():
    cases = load_dataset_from_file("data/adversarial_dataset.json")
    assert len(cases) > 0
    print(f"\n✅ Loaded {len(cases)} adversarial cases")


def test_adversarial_cases_have_correct_category():
    cases = load_dataset_from_file("data/adversarial_dataset.json")
    for case in cases:
        assert case.category == Category.ADVERSARIAL
        assert case.difficulty == Difficulty.HARD
    print(f"\n✅ All {len(cases)} cases correctly tagged as adversarial/hard")


def test_adversarial_cases_have_all_strategies():
    cases = load_dataset_from_file("data/adversarial_dataset.json")
    sources = set(c.source for c in cases)
    print(f"\n✅ Strategies used: {sources}")
    # At least 2 strategies should be present
    assert len(sources) >= 2


def test_generator_produces_valid_cases():
    """Quick smoke test — generate just 3 cases."""
    config = load_config()
    seed_cases = load_dataset_from_file("data/seed_dataset.json")

    generator = AdversarialGenerator(api_key=config.groq_api_key)
    cases = generator.generate(seed_cases=seed_cases[:10], num_adversarial=3)

    assert len(cases) > 0
    for case in cases:
        assert len(case.question) > 0
        assert case.category == Category.ADVERSARIAL
        print(f"\n✅ {case.source}: {case.question[:60]}...")