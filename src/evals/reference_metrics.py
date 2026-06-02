from dataclasses import dataclass
from typing import List
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.utils.data_models import TestCase, EvalResult


@dataclass
class ReferenceScores:
    """All reference-based metric scores for one response."""
    exact_match: float        # 1.0 or 0.0
    f1_score: float           # token overlap F1
    rouge_l: float            # longest common subsequence
    bert_precision: float     # semantic precision
    bert_recall: float        # semantic recall
    bert_f1: float            # semantic F1 (main BERTScore)
    overall: float            # weighted combination

    def to_dict(self) -> dict:
        return {
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "rouge_l": self.rouge_l,
            "bert_precision": self.bert_precision,
            "bert_recall": self.bert_recall,
            "bert_f1": self.bert_f1,
            "overall": self.overall
        }


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fair comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens = set(_normalize(reference).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def _exact_match(prediction: str, reference: str) -> float:
    """Check if prediction exactly matches reference (normalized)."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def _rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return round(scores['rougeL'].fmeasure, 4)


def _bertscore(predictions: List[str], references: List[str]) -> List[dict]:
    """
    Compute BERTScore for a batch of predictions.
    Uses distilbert for speed — still semantically meaningful.
    """
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False
    )
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        results.append({
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4)
        })
    return results


class ReferenceMetricsEval:
    """
    Computes ROUGE-L, BERTScore, Exact Match, and F1
    for a list of pipeline outputs against reference answers.

    These are deterministic — same input always gives same output.
    No API calls needed. Fast and free.
    """

    def score_single(
        self,
        prediction: str,
        reference: str
    ) -> ReferenceScores:
        """Score a single prediction against a reference."""
        em = _exact_match(prediction, reference)
        f1 = _token_f1(prediction, reference)
        rl = _rouge_l(prediction, reference)

        # BERTScore for single item (wrapped in list)
        bs = _bertscore([prediction], [reference])[0]

        # Weighted overall score
        # BERTScore F1 is most meaningful for semantic similarity
        overall = round(
            em * 0.10 +
            f1 * 0.20 +
            rl * 0.30 +
            bs["f1"] * 0.40,
            4
        )

        return ReferenceScores(
            exact_match=em,
            f1_score=f1,
            rouge_l=rl,
            bert_precision=bs["precision"],
            bert_recall=bs["recall"],
            bert_f1=bs["f1"],
            overall=overall
        )

    def score_batch(
        self,
        eval_results: List[EvalResult],
        test_cases: List[TestCase]
    ) -> List[EvalResult]:
        """
        Score a batch of EvalResults and attach reference scores.
        Uses batched BERTScore for efficiency.
        """
        # Build lookup for test cases
        case_map = {c.id: c for c in test_cases}

        predictions = []
        references = []
        valid_indices = []

        for i, result in enumerate(eval_results):
            case = case_map.get(result.test_case_id)
            if case and not result.pipeline_output.startswith("ERROR"):
                predictions.append(result.pipeline_output)
                references.append(case.reference_answer)
                valid_indices.append(i)

        if not predictions:
            return eval_results

        print(f"\n📐 Computing reference metrics for {len(predictions)} results...")

        # Compute all metrics
        em_scores = [_exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [_token_f1(p, r) for p, r in zip(predictions, references)]
        rouge_scores = [_rouge_l(p, r) for p, r in zip(predictions, references)]
        bert_scores = _bertscore(predictions, references)

        # Attach scores back to results
        for idx, i in enumerate(valid_indices):
            bs = bert_scores[idx]
            overall = round(
                em_scores[idx] * 0.10 +
                f1_scores[idx] * 0.20 +
                rouge_scores[idx] * 0.30 +
                bs["f1"] * 0.40,
                4
            )

            eval_results[i].scores["reference"] = ReferenceScores(
                exact_match=em_scores[idx],
                f1_score=f1_scores[idx],
                rouge_l=rouge_scores[idx],
                bert_precision=bs["precision"],
                bert_recall=bs["recall"],
                bert_f1=bs["f1"],
                overall=overall
            ).to_dict()

        avg_bert = sum(b["f1"] for b in bert_scores) / len(bert_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print(f"   Avg ROUGE-L:    {avg_rouge:.3f}")
        print(f"   Avg BERTScore:  {avg_bert:.3f}")

        return eval_results