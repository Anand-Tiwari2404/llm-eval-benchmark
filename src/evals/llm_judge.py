import json
import time
from dataclasses import dataclass
from groq import Groq
from src.utils.data_models import TestCase, EvalResult


JUDGE_PROMPT = """You are an expert AI evaluator. Your job is to score an AI system's response to a question.

## Question
{question}

## Reference Answer (Ground Truth)
{reference_answer}

## AI System's Response
{ai_response}

## Context (if available)
{context}

---

Score the AI response on these 4 dimensions. Be strict and objective.

1. **Correctness** (0-5): Is the answer factually correct compared to the reference?
   - 5: Perfectly correct, matches reference
   - 3: Partially correct, some right elements
   - 1: Mostly wrong but related
   - 0: Completely wrong or hallucinated

2. **Relevance** (0-5): Does the response actually answer the question asked?
   - 5: Directly and completely answers the question
   - 3: Somewhat answers but goes off-topic
   - 1: Barely related to the question
   - 0: Completely irrelevant

3. **Completeness** (0-5): Is the answer complete or missing key information?
   - 5: Complete, nothing important missing
   - 3: Covers main point but missing details
   - 1: Very incomplete
   - 0: Essentially empty or useless

4. **Hallucination** (0-5): Does the response contain made-up information?
   - 5: No hallucinations, all claims are grounded
   - 3: Minor unsupported claims
   - 1: Significant hallucinated content
   - 0: Mostly or completely hallucinated

Respond ONLY with valid JSON, absolutely no other text:
{{
    "correctness": <0-5>,
    "relevance": <0-5>,
    "completeness": <0-5>,
    "hallucination": <0-5>,
    "overall_score": <0-100>,
    "reasoning": "2-3 sentences explaining your scores",
    "verdict": "PASS" or "FAIL"
}}

The overall_score should reflect a weighted combination:
- Correctness: 40%
- Relevance: 25%
- Completeness: 20%
- Hallucination: 15%
Multiply the weighted sum by 20 to get a 0-100 score.

verdict is PASS if overall_score >= 60, FAIL otherwise."""


@dataclass
class JudgeScore:
    """Structured output from the LLM judge."""
    correctness: float
    relevance: float
    completeness: float
    hallucination: float
    overall_score: float
    reasoning: str
    verdict: str
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "hallucination": self.hallucination,
            "overall_score": self.overall_score,
            "reasoning": self.reasoning,
            "verdict": self.verdict
        }


class LLMJudge:
    """
    Uses Groq (llama-3.3-70b) as a judge to score pipeline responses.
    Larger model as judge, smaller model as pipeline = realistic eval setup.
    Caches results to avoid redundant API calls.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self._cache: dict = {}

    def _build_prompt(self, test_case: TestCase, ai_response: str) -> str:
        return JUDGE_PROMPT.format(
            question=test_case.question,
            reference_answer=test_case.reference_answer,
            ai_response=ai_response,
            context=test_case.context or "No context provided"
        )

    def _parse_response(self, raw: str) -> JudgeScore:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        data = json.loads(cleaned)

        return JudgeScore(
            correctness=float(data.get("correctness", 0)),
            relevance=float(data.get("relevance", 0)),
            completeness=float(data.get("completeness", 0)),
            hallucination=float(data.get("hallucination", 0)),
            overall_score=float(data.get("overall_score", 0)),
            reasoning=data.get("reasoning", ""),
            verdict=data.get("verdict", "FAIL"),
            raw_response=raw
        )

    def score(self, test_case: TestCase, ai_response: str) -> JudgeScore:
        """Score a single pipeline response with retry logic."""
        cache_key = f"{test_case.id}:{hash(ai_response)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(test_case, ai_response)
        score = None

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512
                )
                raw = response.choices[0].message.content.strip()
                score = self._parse_response(raw)
                break
            except json.JSONDecodeError:
                score = JudgeScore(
                    correctness=0, relevance=0, completeness=0,
                    hallucination=0, overall_score=0,
                    reasoning="Judge returned invalid JSON",
                    verdict="FAIL"
                )
                break
            except Exception as e:
                error_msg = str(e)
                if attempt < 2:
                    time.sleep(3)
                    continue
                score = JudgeScore(
                    correctness=0, relevance=0, completeness=0,
                    hallucination=0, overall_score=0,
                    reasoning=f"Judge error: {error_msg[:100]}",
                    verdict="FAIL"
                )
                break

        if score is None:
            score = JudgeScore(
                correctness=0, relevance=0, completeness=0,
                hallucination=0, overall_score=0,
                reasoning="All retry attempts failed",
                verdict="FAIL"
            )

        self._cache[cache_key] = score
        return score

    def score_eval_result(self, test_case: TestCase, eval_result: EvalResult) -> EvalResult:
        """Score an EvalResult and attach scores to it."""
        judge_score = self.score(test_case, eval_result.pipeline_output)
        eval_result.scores["judge"] = judge_score.to_dict()
        eval_result.metadata["verdict"] = judge_score.verdict
        return eval_result