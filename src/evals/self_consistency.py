import time
from dataclasses import dataclass
from typing import List
from groq import Groq
from src.utils.data_models import TestCase


@dataclass
class ConsistencyResult:
    """Result of running self-consistency check on one test case."""
    test_case_id: str
    question: str
    samples: List[str]           # all 5 answers collected
    consistency_score: float     # 0.0 to 1.0
    is_consistent: bool          # True if score >= threshold
    majority_answer: str         # most common answer
    reasoning: str               # explanation of the score

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "samples": self.samples,
            "consistency_score": self.consistency_score,
            "is_consistent": self.is_consistent,
            "majority_answer": self.majority_answer,
            "reasoning": self.reasoning
        }


class SelfConsistencyChecker:
    """
    Detects hallucinations by running the same question multiple times
    at high temperature and checking if answers agree.

    High consistency = model knows the answer confidently.
    Low consistency  = model is guessing = likely hallucination.
    """

    def __init__(
        self,
        api_key: str,
        pipeline_model: str = "llama-3.1-8b-instant",
        judge_model: str = "llama-3.3-70b-versatile",
        num_samples: int = 5,
        temperature: float = 0.8,
        consistency_threshold: float = 0.6
    ):
        self.client = Groq(api_key=api_key)
        self.pipeline_model = pipeline_model
        self.judge_model = judge_model
        self.num_samples = num_samples
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

    def _sample_answers(self, question: str, context: str = None) -> List[str]:
        """Run the question num_samples times at high temperature."""
        prompt = f"Context: {context}\n\nQuestion: {question}" if context else question

        samples = []
        for _ in range(self.num_samples):
            try:
                response = self.client.chat.completions.create(
                    model=self.pipeline_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=150
                )
                answer = response.choices[0].message.content.strip()
                samples.append(answer)
                time.sleep(0.5)  # small delay between samples
            except Exception as e:
                samples.append(f"ERROR: {str(e)[:50]}")

        return samples

    def _check_agreement(self, question: str, samples: List[str]) -> dict:
        """
        Use the judge LLM to check if the samples agree with each other.
        Returns agreement score and majority answer.
        """
        samples_text = "\n".join(
            f"Answer {i+1}: {s}" for i, s in enumerate(samples)
        )

        prompt = f"""You are evaluating whether multiple AI responses to the same question are consistent with each other.

Question: {question}

Here are {len(samples)} different responses:
{samples_text}

Analyze these responses and respond ONLY with valid JSON:
{{
    "agreement_score": <0.0 to 1.0, where 1.0 means all answers say the same thing>,
    "majority_answer": "<the most common/consistent answer across responses>",
    "reasoning": "<1-2 sentences explaining the consistency level>",
    "consistent_count": <how many answers agree with the majority>
}}

Scoring guide:
- 1.0: All answers are identical or semantically equivalent
- 0.8: Most answers agree, minor wording differences
- 0.6: More than half agree on the core answer
- 0.4: Answers are split between 2 different answers
- 0.2: Most answers disagree with each other
- 0.0: Every answer is completely different"""

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256
            )
            raw = response.choices[0].message.content.strip()

            # Clean JSON
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            import json
            data = json.loads(raw)
            return data

        except Exception as e:
            return {
                "agreement_score": 0.0,
                "majority_answer": samples[0] if samples else "",
                "reasoning": f"Agreement check failed: {str(e)[:80]}",
                "consistent_count": 0
            }

    def check(self, test_case: TestCase) -> ConsistencyResult:
        """
        Run full self-consistency check on a single test case.
        """
        # Step 1: collect multiple samples
        samples = self._sample_answers(
            question=test_case.question,
            context=test_case.context
        )

        # Step 2: check agreement between samples
        agreement = self._check_agreement(test_case.question, samples)

        consistency_score = float(agreement.get("agreement_score", 0.0))
        is_consistent = consistency_score >= self.consistency_threshold

        return ConsistencyResult(
            test_case_id=test_case.id,
            question=test_case.question,
            samples=samples,
            consistency_score=round(consistency_score, 2),
            is_consistent=is_consistent,
            majority_answer=agreement.get("majority_answer", ""),
            reasoning=agreement.get("reasoning", "")
        )

    def check_batch(
        self,
        test_cases: List[TestCase],
        max_cases: int = None
    ) -> List[ConsistencyResult]:
        """Run consistency check on a list of test cases."""
        cases = test_cases[:max_cases] if max_cases else test_cases
        results = []

        print(f"\n🔄 Running self-consistency check on {len(cases)} cases...")
        print(f"   Samples per question: {self.num_samples}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Consistency threshold: {self.consistency_threshold}\n")

        consistent = 0
        for i, case in enumerate(cases):
            print(f"  [{i+1}/{len(cases)}] {case.id}...", end=" ", flush=True)
            result = self.check(case)
            results.append(result)

            if result.is_consistent:
                consistent += 1
                print(f"✅ {result.consistency_score:.2f}")
            else:
                print(f"⚠️  {result.consistency_score:.2f} — possible hallucination")

        print(f"\n📊 Consistency summary:")
        print(f"   Consistent:   {consistent}/{len(cases)}")
        print(f"   Inconsistent: {len(cases)-consistent}/{len(cases)}")
        avg = sum(r.consistency_score for r in results) / len(results)
        print(f"   Avg score:    {avg:.2f}")

        return results