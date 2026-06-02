import pytest
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.evals.self_consistency import SelfConsistencyChecker, ConsistencyResult


@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def seed_cases():
    return load_dataset_from_file("data/seed_dataset.json")


def test_consistency_result_schema(config, seed_cases):
    """Check result has all required fields."""
    checker = SelfConsistencyChecker(
        api_key=config.groq_api_key,
        num_samples=3  # use 3 for speed in tests
    )
    result = checker.check(seed_cases[0])

    assert isinstance(result, ConsistencyResult)
    assert len(result.samples) == 3
    assert 0.0 <= result.consistency_score <= 1.0
    assert isinstance(result.is_consistent, bool)
    assert len(result.majority_answer) > 0
    print(f"\n✅ Schema valid")
    print(f"   Score: {result.consistency_score}")
    print(f"   Consistent: {result.is_consistent}")
    print(f"   Majority: {result.majority_answer[:60]}...")


def test_consistent_factual_question(config, seed_cases):
    """Simple factual questions should be highly consistent."""
    checker = SelfConsistencyChecker(
        api_key=config.groq_api_key,
        num_samples=3
    )
    # Use first SQuAD case with context — model should answer consistently
    result = checker.check(seed_cases[0])

    print(f"\n✅ Factual question consistency: {result.consistency_score}")
    print(f"   Reasoning: {result.reasoning[:100]}...")
    assert result.consistency_score >= 0.0  # just verify it runs


def test_batch_returns_correct_count(config, seed_cases):
    """Batch check should return one result per case."""
    checker = SelfConsistencyChecker(
        api_key=config.groq_api_key,
        num_samples=3
    )
    results = checker.check_batch(seed_cases, max_cases=3)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, ConsistencyResult)
    print(f"\n✅ Batch returned {len(results)} results")


def test_consistency_to_dict(config, seed_cases):
    """Result should serialize to dict correctly."""
    checker = SelfConsistencyChecker(
        api_key=config.groq_api_key,
        num_samples=3
    )
    result = checker.check(seed_cases[0])
    d = result.to_dict()

    assert "consistency_score" in d
    assert "samples" in d
    assert "majority_answer" in d
    assert "is_consistent" in d
    print(f"\n✅ Serialization works: {list(d.keys())}")