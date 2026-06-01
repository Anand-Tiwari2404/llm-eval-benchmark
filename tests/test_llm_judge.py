import pytest
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.evals.llm_judge import LLMJudge, JudgeScore
from src.evals.eval_runner import EvalRunner
from src.pipelines.groq_pipeline import GroqPipeline


@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def judge(config):
    return LLMJudge(api_key=config.groq_api_key)

@pytest.fixture
def seed_cases():
    return load_dataset_from_file("data/seed_dataset.json")


def test_judge_scores_correct_answer(judge, seed_cases):
    """A correct answer should score high."""
    case = seed_cases[0]
    score = judge.score(case, case.reference_answer)

    assert isinstance(score, JudgeScore)
    assert score.overall_score >= 60
    assert score.verdict == "PASS"
    print(f"\n✅ Correct answer scored: {score.overall_score}/100")
    print(f"   Reasoning: {score.reasoning[:100]}...")


def test_judge_scores_wrong_answer_low(judge, seed_cases):
    """A completely wrong answer should score low."""
    case = seed_cases[0]
    score = judge.score(case, "I have no idea, probably something random.")

    assert isinstance(score, JudgeScore)
    assert score.overall_score < 60
    assert score.verdict == "FAIL"
    print(f"\n✅ Wrong answer correctly scored low: {score.overall_score}/100")


def test_judge_returns_all_dimensions(judge, seed_cases):
    """Judge must return all 4 scoring dimensions."""
    case = seed_cases[0]
    score = judge.score(case, "Paris is the capital of France.")

    assert hasattr(score, "correctness")
    assert hasattr(score, "relevance")
    assert hasattr(score, "completeness")
    assert hasattr(score, "hallucination")
    assert 0 <= score.overall_score <= 100
    print(f"\n✅ All dimensions present")
    print(f"   Correctness:  {score.correctness}/5")
    print(f"   Relevance:    {score.relevance}/5")
    print(f"   Completeness: {score.completeness}/5")
    print(f"   Hallucination:{score.hallucination}/5")


def test_eval_runner_end_to_end(config, seed_cases):
    """Full pipeline → judge flow on 3 cases."""
    pipeline = GroqPipeline(api_key=config.groq_api_key)
    judge = LLMJudge(api_key=config.groq_api_key)
    runner = EvalRunner(pipeline=pipeline, judge=judge)

    results = runner.run(test_cases=seed_cases[:3], verbose=True)

    assert len(results) == 3
    for result in results:
        assert "judge" in result.scores
        assert result.scores["judge"]["overall_score"] >= 0
        print(f"\n✅ {result.test_case_id}: "
              f"{result.scores['judge']['overall_score']}/100 "
              f"— {result.metadata['verdict']}")