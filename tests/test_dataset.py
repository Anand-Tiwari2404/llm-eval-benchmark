import pytest
import os
from src.utils.data_models import TestCase, Category, Difficulty
from src.utils.dataset_loader import load_dataset_from_file


def test_seed_dataset_exists():
    assert os.path.exists("data/seed_dataset.json"), \
        "Run build_dataset.py first!"
    print("\n✅ seed_dataset.json exists")


def test_dataset_loads_correctly():
    cases = load_dataset_from_file("data/seed_dataset.json")
    assert len(cases) > 0
    print(f"\n✅ Loaded {len(cases)} test cases")


def test_dataset_has_correct_structure():
    cases = load_dataset_from_file("data/seed_dataset.json")
    case = cases[0]

    assert isinstance(case, TestCase)
    assert len(case.id) > 0
    assert len(case.question) > 0
    assert len(case.reference_answer) > 0
    assert isinstance(case.category, Category)
    assert isinstance(case.difficulty, Difficulty)
    print(f"\n✅ Structure valid: {case.id}")
    print(f"   Q: {case.question[:60]}...")
    print(f"   A: {case.reference_answer}")


def test_dataset_has_both_sources():
    cases = load_dataset_from_file("data/seed_dataset.json")
    sources = set(c.source for c in cases)
    assert "squad" in sources
    assert "openbookqa" in sources
    print(f"\n✅ Sources found: {sources}")


def test_dataset_distribution():
    cases = load_dataset_from_file("data/seed_dataset.json")
    comprehension = [c for c in cases if c.category == Category.COMPREHENSION]
    reasoning = [c for c in cases if c.category == Category.REASONING]
    print(f"\n✅ Distribution:")
    print(f"   Comprehension (SQuAD):      {len(comprehension)}")
    print(f"   Reasoning (OpenBookQA):     {len(reasoning)}")
    assert len(comprehension) > 0
    assert len(reasoning) > 0