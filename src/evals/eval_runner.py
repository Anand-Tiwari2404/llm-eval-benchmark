import time
from typing import List, Optional
from unittest import case
from tqdm import tqdm
from src.utils.data_models import TestCase, EvalResult
from src.pipelines.base import BasePipeline
from src.evals.llm_judge import LLMJudge


class EvalRunner:
    """
    Runs a pipeline against a list of test cases,
    scores each response using the LLM judge,
    and returns a list of EvalResults.
    """

    def __init__(self, pipeline: BasePipeline, judge: LLMJudge):
        self.pipeline = pipeline
        self.judge = judge

    def run(
        self,
        test_cases: List[TestCase],
        max_cases: Optional[int] = None,
        verbose: bool = True
    ) -> List[EvalResult]:
        """
        Run eval on a list of test cases.
        Returns list of EvalResult with judge scores attached.
        """
        cases = test_cases[:max_cases] if max_cases else test_cases
        results = []
        passed = 0
        failed = 0

        if verbose:
            print(f"\n🚀 Running eval: {self.pipeline.model}")
            print(f"   Test cases: {len(cases)}")
            print(f"   Judge: {self.judge.model}\n")

        for case in tqdm(cases, desc=f"Evaluating"):
            # Step 1: run the pipeline
            # For SQuAD-style questions, include context in the prompt
            if case.context:
                prompt = f"Context: {case.context}\n\nQuestion: {case.question}"
            else:
                prompt = case.question
            pipeline_response = self.pipeline.run(prompt)

            # Step 2: build EvalResult
            eval_result = EvalResult(
                test_case_id=case.id,
                pipeline_model=self.pipeline.model,
                pipeline_output=pipeline_response.output,
                latency_seconds=pipeline_response.latency_seconds,
                cost_usd=pipeline_response.cost_usd,
                input_tokens=pipeline_response.input_tokens,
                output_tokens=pipeline_response.output_tokens,
                metadata={
                    "category": case.category.value,
                    "difficulty": case.difficulty.value,
                    "source": case.source
                }
            )

            # Step 3: judge the response
            eval_result = self.judge.score_eval_result(case, eval_result)
            time.sleep(6)

            # Track pass/fail
            if eval_result.metadata.get("verdict") == "PASS":
                passed += 1
            else:
                failed += 1

            results.append(eval_result)

        if verbose:
            total = len(results)
            pass_rate = (passed / total * 100) if total > 0 else 0
            print(f"\n📊 Eval complete!")
            print(f"   Passed: {passed}/{total} ({pass_rate:.1f}%)")
            print(f"   Failed: {failed}/{total}")
            avg_score = sum(
                r.scores.get("judge", {}).get("overall_score", 0)
                for r in results
            ) / total if total > 0 else 0
            print(f"   Avg judge score: {avg_score:.1f}/100")

        return results