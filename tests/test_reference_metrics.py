import pytest
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.utils.data_models import EvalResult
from src.evals.reference_metrics import ReferenceMetricsEval, ReferenceScores


@pytest.fixture
def metrics():
    return ReferenceMetricsEval()

@pytest.fixture
def seed_cases():
    return load_dataset_from_file("data/seed_dataset.json")


def test_exact_match_perfect(metrics):
    """Identical strings should score 1.0 exact match."""
    case = type('obj', (object,), {
        'id': 'test_1',
        'question': 'test',
        'reference_answer': 'Saint Bernadette Soubirous',
        'context': None
    })()
    scores = metrics.score_single(
        "Saint Bernadette Soubirous",
        "Saint Bernadette Soubirous"
    )
    assert scores.exact_match == 1.0
    print(f"\n✅ Exact match: {scores.exact_match}")


def test_exact_match_wrong(metrics):
    """Wrong answer should score 0.0 exact match."""
    scores = metrics.score_single("Paris", "London")
    assert scores.exact_match == 0.0
    print(f"\n✅ Wrong answer exact match: {scores.exact_match}")


def test_bertscore_semantic_similarity(metrics):
    """Semantically similar answers should score high BERTScore."""
    scores = metrics.score_single(
        "The Virgin Mary appeared to Bernadette",
        "Saint Bernadette Soubirous"
    )
    assert scores.bert_f1 > 0.7
    print(f"\n✅ Semantic similarity BERTScore: {scores.bert_f1}")


def test_rouge_l_partial_match(metrics):
    """Partial word overlap should give partial ROUGE score."""
    scores = metrics.score_single(
        "Saint Bernadette Soubirous saw the Virgin Mary",
        "Saint Bernadette Soubirous"
    )
    assert 0.0 < scores.rouge_l <= 1.0
    print(f"\n✅ ROUGE-L partial match: {scores.rouge_l}")


def test_all_metrics_present(metrics):
    """All metric fields must be present in output."""
    scores = metrics.score_single("Paris", "Paris is the capital of France")
    d = scores.to_dict()

    assert "exact_match" in d
    assert "f1_score" in d
    assert "rouge_l" in d
    assert "bert_f1" in d
    assert "overall" in d
    assert 0.0 <= d["overall"] <= 1.0
    print(f"\n✅ All metrics present: {list(d.keys())}")
    print(f"   Overall: {d['overall']}")


def test_batch_scoring(metrics, seed_cases):
    """Batch scoring should attach reference scores to all results."""
    # Create mock eval results
    mock_results = [
        EvalResult(
            test_case_id=case.id,
            pipeline_model="test",
            pipeline_output=case.reference_answer,  # perfect answers
            latency_seconds=0.1,
            cost_usd=0.0,
            input_tokens=10,
            output_tokens=10
        )
        for case in seed_cases[:3]
    ]

    results = metrics.score_batch(mock_results, seed_cases[:3])

    for result in results:
        assert "reference" in result.scores
        assert result.scores["reference"]["exact_match"] == 1.0
        print(f"\n✅ {result.test_case_id}: overall={result.scores['reference']['overall']}")