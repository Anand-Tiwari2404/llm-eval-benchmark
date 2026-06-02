from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.pipelines.groq_pipeline import GroqPipeline
from src.evals.llm_judge import LLMJudge
from src.evals.eval_runner import EvalRunner
from src.evals.reference_metrics import ReferenceMetricsEval

if __name__ == "__main__":
    config = load_config()

    # Load 5 cases
    all_cases = load_dataset_from_file("data/seed_dataset.json")
    demo_cases = all_cases[:5]

    # Run pipeline + judge
    pipeline = GroqPipeline(api_key=config.groq_api_key)
    judge = LLMJudge(api_key=config.groq_api_key)
    runner = EvalRunner(pipeline=pipeline, judge=judge)
    results = runner.run(test_cases=demo_cases, verbose=True)

    # Now add reference metrics on top
    metrics_eval = ReferenceMetricsEval()
    results = metrics_eval.score_batch(results, demo_cases)

    # Show combined scores
    print("\n🔍 Combined scores (judge + reference metrics):")
    for result in results:
        judge_scores = result.scores.get("judge", {})
        ref_scores = result.scores.get("reference", {})

        print(f"\n  Case: {result.test_case_id}")
        print(f"  Output: {result.pipeline_output[:60]}...")
        print(f"  ── Judge scores ──")
        print(f"     Overall:     {judge_scores.get('overall_score')}/100")
        print(f"     Verdict:     {result.metadata.get('verdict')}")
        print(f"  ── Reference metrics ──")
        print(f"     Exact Match: {ref_scores.get('exact_match')}")
        print(f"     F1:          {ref_scores.get('f1_score')}")
        print(f"     ROUGE-L:     {ref_scores.get('rouge_l')}")
        print(f"     BERTScore:   {ref_scores.get('bert_f1')}")
        print(f"     Overall:     {ref_scores.get('overall')}")