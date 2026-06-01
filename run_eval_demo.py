from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.pipelines.groq_pipeline import GroqPipeline
from src.evals.llm_judge import LLMJudge
from src.evals.eval_runner import EvalRunner

if __name__ == "__main__":
    config = load_config()

    # Load 10 test cases for the demo
    all_cases = load_dataset_from_file("data/seed_dataset.json")
    demo_cases = all_cases[:10]

    # Pipeline: small fast model (llama-3.1-8b)
    # Judge:    large capable model (llama-3.3-70b)
    # This is the correct eval setup — judge > pipeline
    pipeline = GroqPipeline(api_key=config.groq_api_key)
    judge = LLMJudge(api_key=config.groq_api_key)
    runner = EvalRunner(pipeline=pipeline, judge=judge)

    # Run eval
    results = runner.run(test_cases=demo_cases, verbose=True)

    # Show detailed results for first 3
    print("\n🔍 Detailed results (first 3):")
    for result in results[:3]:
        scores = result.scores.get("judge", {})
        print(f"\n  Case: {result.test_case_id}")
        print(f"  Output: {result.pipeline_output[:80]}...")
        print(f"  Verdict: {result.metadata.get('verdict')}")
        print(f"  Overall: {scores.get('overall_score')}/100")
        print(f"  Correctness:  {scores.get('correctness')}/5")
        print(f"  Relevance:    {scores.get('relevance')}/5")
        print(f"  Completeness: {scores.get('completeness')}/5")
        print(f"  Hallucination:{scores.get('hallucination')}/5")
        print(f"  Reasoning: {scores.get('reasoning', '')[:120]}...")