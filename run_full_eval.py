import uuid
from datetime import datetime
from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.utils.aggregator import aggregate_results
from src.utils.regression_store import (
    save_report, load_all_runs, regression_diff, get_latest_run_id
)
from src.pipelines.groq_pipeline import GroqPipeline
from src.evals.llm_judge import LLMJudge
from src.evals.eval_runner import EvalRunner
from src.evals.reference_metrics import ReferenceMetricsEval


def run_full_eval(num_cases: int = 10):
    config = load_config()
    run_id = f"run-{str(uuid.uuid4())[:8]}"
    timestamp = datetime.now().isoformat()

    print(f"\n{'='*55}")
    print(f"  LLM EVAL BENCHMARK — Full Run")
    print(f"  Run ID:    {run_id}")
    print(f"  Timestamp: {timestamp}")
    print(f"{'='*55}\n")

    # Load test cases
    all_cases = load_dataset_from_file("data/seed_dataset.json")
    test_cases = all_cases[:num_cases]
    print(f"📂 Loaded {len(test_cases)} test cases\n")

    # Step 1: Run pipeline + LLM judge
    print("── Step 1: LLM-as-judge eval ──")
    pipeline = GroqPipeline(api_key=config.groq_api_key)
    judge = LLMJudge(api_key=config.groq_api_key)
    runner = EvalRunner(pipeline=pipeline, judge=judge)
    results = runner.run(test_cases=test_cases, verbose=True)

    # Step 2: Reference metrics
    print("\n── Step 2: Reference metrics ──")
    metrics_eval = ReferenceMetricsEval()
    results = metrics_eval.score_batch(results, test_cases)

    # Step 3: Aggregate
    print("\n── Step 3: Aggregating scores ──")
    report = aggregate_results(
        results=results,
        run_id=run_id,
        pipeline_model=pipeline.model,
        timestamp=timestamp
    )

    # Step 4: Save to database
    print("\n── Step 4: Saving to database ──")
    save_report(report)

    # Step 5: Regression check
    print("\n── Step 5: Regression check ──")
    baseline_id = get_latest_run_id()
    if baseline_id:
        diff = regression_diff(report, baseline_id)
        if diff.get("has_regressions"):
            print(f"\n🚨 REGRESSIONS DETECTED vs {baseline_id}:")
            for reg in diff["regressions"]:
                print(f"   ▼ {reg['metric']}: "
                      f"{reg['baseline']} → {reg['current']} "
                      f"({reg['delta_pct']}%)")
        else:
            print(f"✅ No regressions vs {baseline_id}")

        if diff.get("improvements"):
            print(f"\n📈 Improvements:")
            for imp in diff["improvements"]:
                print(f"   ▲ {imp['metric']}: "
                      f"{imp['baseline']} → {imp['current']} "
                      f"(+{imp['delta_pct']}%)")
    else:
        print("ℹ️  No baseline run found — this is the first run")

    # Final summary
    print(f"\n{'='*55}")
    print(f"  FINAL REPORT — {run_id}")
    print(f"{'='*55}")
    print(f"  Pipeline:       {report.pipeline_model}")
    print(f"  Cases:          {report.total_cases}")
    print(f"  Judge pass rate:{report.judge_pass_rate}%")
    print(f"  Avg judge score:{report.judge_avg_score}/100")
    print(f"  ROUGE-L:        {report.avg_rouge_l}")
    print(f"  BERTScore F1:   {report.avg_bert_f1}")
    print(f"  Exact match:    {report.avg_exact_match}")
    print(f"  Total cost:     ${report.total_cost_usd}")
    print(f"  Avg latency:    {report.avg_latency_seconds}s")
    print(f"\n  Category breakdown:")
    for cat, score in report.category_scores.items():
        print(f"    {cat}: {score}/100")
    print(f"{'='*55}\n")

    return report


if __name__ == "__main__":
    run_full_eval(num_cases=10)