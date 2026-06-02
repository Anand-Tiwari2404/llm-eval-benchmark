import pytest
import os
from datetime import datetime
from src.utils.data_models import EvalResult
from src.utils.aggregator import aggregate_results, EvalReport
from src.utils.regression_store import (
    init_db, save_report, load_all_runs,
    regression_diff, DB_PATH
)

TEST_DB = "outputs/test_eval_runs.db"


def make_mock_result(id: str, score: float, verdict: str) -> EvalResult:
    result = EvalResult(
        test_case_id=id,
        pipeline_model="test-model",
        pipeline_output="Test output",
        latency_seconds=0.5,
        cost_usd=0.001,
        input_tokens=100,
        output_tokens=50,
        metadata={"verdict": verdict, "category": "factual"}
    )
    result.scores["judge"] = {
        "overall_score": score,
        "correctness": score / 20,
        "relevance": score / 20,
        "completeness": score / 20,
        "hallucination": score / 20,
        "reasoning": "test",
        "verdict": verdict
    }
    result.scores["reference"] = {
        "exact_match": 1.0 if score > 80 else 0.0,
        "f1_score": score / 100,
        "rouge_l": score / 100,
        "bert_f1": score / 100,
        "bert_precision": score / 100,
        "bert_recall": score / 100,
        "overall": score / 100
    }
    return result


def test_aggregate_results():
    results = [
        make_mock_result("case_1", 90.0, "PASS"),
        make_mock_result("case_2", 80.0, "PASS"),
        make_mock_result("case_3", 40.0, "FAIL"),
    ]
    report = aggregate_results(
        results=results,
        run_id="test-run-001",
        pipeline_model="test-model",
        timestamp=datetime.now().isoformat()
    )
    assert isinstance(report, EvalReport)
    assert report.total_cases == 3
    assert report.judge_pass_rate == pytest.approx(66.67, abs=0.1)
    assert report.judge_avg_score == pytest.approx(70.0, abs=0.1)
    print(f"\n✅ Aggregation correct: {report.judge_avg_score}/100 avg")


def test_save_and_load_report():
    results = [make_mock_result("case_1", 85.0, "PASS")]
    report = aggregate_results(
        results=results,
        run_id="test-run-save",
        pipeline_model="test-model",
        timestamp=datetime.now().isoformat()
    )
    save_report(report, db_path=TEST_DB)
    runs = load_all_runs(db_path=TEST_DB)
    run_ids = [r["run_id"] for r in runs]
    assert "test-run-save" in run_ids
    print(f"\n✅ Saved and loaded: {run_ids}")


def test_regression_detection():
    """Current run worse than baseline → regression flagged."""
    # Save baseline (high scores)
    baseline_results = [make_mock_result("case_1", 90.0, "PASS")]
    baseline = aggregate_results(
        results=baseline_results,
        run_id="baseline-run",
        pipeline_model="test-model",
        timestamp="2024-01-01T00:00:00"
    )
    save_report(baseline, db_path=TEST_DB)

    # Current run (lower scores = regression)
    current_results = [make_mock_result("case_1", 50.0, "FAIL")]
    current = aggregate_results(
        results=current_results,
        run_id="current-run",
        pipeline_model="test-model",
        timestamp=datetime.now().isoformat()
    )

    diff = regression_diff(current, "baseline-run", db_path=TEST_DB)
    assert diff["has_regressions"] is True
    print(f"\n✅ Regression detected: {len(diff['regressions'])} metrics")
    for reg in diff["regressions"]:
        print(f"   ▼ {reg['metric']}: {reg['delta_pct']}%")


def test_improvement_detection():
    """Current run better than baseline → improvement flagged."""
    low_results = [make_mock_result("case_1", 40.0, "FAIL")]
    low = aggregate_results(
        results=low_results,
        run_id="low-run",
        pipeline_model="test-model",
        timestamp="2024-01-01T00:00:00"
    )
    save_report(low, db_path=TEST_DB)

    high_results = [make_mock_result("case_1", 95.0, "PASS")]
    high = aggregate_results(
        results=high_results,
        run_id="high-run",
        pipeline_model="test-model",
        timestamp=datetime.now().isoformat()
    )

    diff = regression_diff(high, "low-run", db_path=TEST_DB)
    assert len(diff["improvements"]) > 0
    print(f"\n✅ Improvement detected: {len(diff['improvements'])} metrics")
    for imp in diff["improvements"]:
        print(f"   ▲ {imp['metric']}: +{imp['delta_pct']}%")