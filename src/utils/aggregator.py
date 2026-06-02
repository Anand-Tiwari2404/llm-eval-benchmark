from dataclasses import dataclass, field
from typing import List, Optional
from src.utils.data_models import EvalResult


@dataclass
class EvalReport:
    """
    Aggregated report from one full eval run.
    Contains summary stats across all eval results.
    """
    run_id: str
    pipeline_model: str
    total_cases: int
    timestamp: str

    # Judge scores
    judge_pass_rate: float        # 0-100
    judge_avg_score: float        # 0-100
    judge_avg_correctness: float  # 0-5
    judge_avg_relevance: float    # 0-5
    judge_avg_completeness: float # 0-5
    judge_avg_hallucination: float# 0-5

    # Self-consistency
    consistency_avg: float        # 0-1
    consistency_rate: float       # % consistent

    # Reference metrics
    avg_rouge_l: float            # 0-1
    avg_bert_f1: float            # 0-1
    avg_exact_match: float        # 0-1
    avg_f1: float                 # 0-1

    # Cost + performance
    total_cost_usd: float
    avg_latency_seconds: float
    total_cases_failed: int

    # Per-category breakdown
    category_scores: dict = field(default_factory=dict)

    # Raw results
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "pipeline_model": self.pipeline_model,
            "total_cases": self.total_cases,
            "timestamp": self.timestamp,
            "judge_pass_rate": self.judge_pass_rate,
            "judge_avg_score": self.judge_avg_score,
            "judge_avg_correctness": self.judge_avg_correctness,
            "judge_avg_relevance": self.judge_avg_relevance,
            "judge_avg_completeness": self.judge_avg_completeness,
            "judge_avg_hallucination": self.judge_avg_hallucination,
            "consistency_avg": self.consistency_avg,
            "consistency_rate": self.consistency_rate,
            "avg_rouge_l": self.avg_rouge_l,
            "avg_bert_f1": self.avg_bert_f1,
            "avg_exact_match": self.avg_exact_match,
            "avg_f1": self.avg_f1,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_seconds": self.avg_latency_seconds,
            "total_cases_failed": self.total_cases_failed,
            "category_scores": self.category_scores
        }


def aggregate_results(
    results: List[EvalResult],
    run_id: str,
    pipeline_model: str,
    timestamp: str
) -> EvalReport:
    """
    Aggregate a list of EvalResults into a single EvalReport.
    Computes averages across all eval dimensions.
    """
    total = len(results)
    if total == 0:
        raise ValueError("No results to aggregate")

    # Judge scores
    judge_scores = [
        r.scores.get("judge", {}) for r in results
        if "judge" in r.scores
    ]
    judge_pass = sum(
        1 for r in results
        if r.metadata.get("verdict") == "PASS"
    )

    judge_avg_score = _avg([s.get("overall_score", 0) for s in judge_scores])
    judge_pass_rate = round(judge_pass / total * 100, 2)
    judge_avg_correctness = _avg([s.get("correctness", 0) for s in judge_scores])
    judge_avg_relevance = _avg([s.get("relevance", 0) for s in judge_scores])
    judge_avg_completeness = _avg([s.get("completeness", 0) for s in judge_scores])
    judge_avg_hallucination = _avg([s.get("hallucination", 0) for s in judge_scores])

    # Reference metrics
    ref_scores = [
        r.scores.get("reference", {}) for r in results
        if "reference" in r.scores
    ]
    avg_rouge_l = _avg([s.get("rouge_l", 0) for s in ref_scores])
    avg_bert_f1 = _avg([s.get("bert_f1", 0) for s in ref_scores])
    avg_exact_match = _avg([s.get("exact_match", 0) for s in ref_scores])
    avg_f1 = _avg([s.get("f1_score", 0) for s in ref_scores])

    # Consistency scores
    consistency_scores = [
        r.scores.get("consistency", {}) for r in results
        if "consistency" in r.scores
    ]
    consistency_avg = _avg([s.get("score", 0) for s in consistency_scores])
    consistency_rate = _avg([
        1.0 if s.get("is_consistent", False) else 0.0
        for s in consistency_scores
    ]) * 100

    # Cost + latency
    total_cost = sum(r.cost_usd for r in results)
    avg_latency = _avg([r.latency_seconds for r in results])
    total_failed = sum(
        1 for r in results
        if r.metadata.get("verdict") == "FAIL"
    )

    # Per-category breakdown
    category_scores = {}
    for result in results:
        cat = result.metadata.get("category", "unknown")
        if cat not in category_scores:
            category_scores[cat] = []
        score = result.scores.get("judge", {}).get("overall_score", 0)
        category_scores[cat].append(score)

    category_averages = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in category_scores.items()
    }

    return EvalReport(
        run_id=run_id,
        pipeline_model=pipeline_model,
        total_cases=total,
        timestamp=timestamp,
        judge_pass_rate=judge_pass_rate,
        judge_avg_score=judge_avg_score,
        judge_avg_correctness=judge_avg_correctness,
        judge_avg_relevance=judge_avg_relevance,
        judge_avg_completeness=judge_avg_completeness,
        judge_avg_hallucination=judge_avg_hallucination,
        consistency_avg=consistency_avg,
        consistency_rate=consistency_rate,
        avg_rouge_l=avg_rouge_l,
        avg_bert_f1=avg_bert_f1,
        avg_exact_match=avg_exact_match,
        avg_f1=avg_f1,
        total_cost_usd=round(total_cost, 6),
        avg_latency_seconds=round(avg_latency, 3),
        total_cases_failed=total_failed,
        category_scores=category_averages,
        results=results
    )


def _avg(values: list) -> float:
    """Safe average that handles empty lists."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)