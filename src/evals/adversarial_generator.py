import json
import uuid
from typing import List
from groq import Groq
from tqdm import tqdm
from src.utils.data_models import TestCase, Category, Difficulty


# Prompts for each adversarial strategy
ADVERSARIAL_PROMPTS = {
    "false_premise": """You are an expert at creating tricky questions that test AI robustness.

Take this question and rewrite it with a subtle FALSE PREMISE embedded in it.
The false premise should be believable but incorrect.

Original question: {question}
Original answer: {answer}

Rules:
- Keep the question on the same topic
- Embed exactly one false premise
- Make it sound natural and convincing
- The correct response should point out the false premise

Respond ONLY with valid JSON, no extra text:
{{
    "adversarial_question": "your rewritten question here",
    "false_premise": "what false assumption is embedded",
    "correct_response": "what a good AI should say"
}}""",

    "misleading_context": """You are an expert at creating tricky questions that test AI robustness.

Take this question and add MISLEADING CONTEXT that might lead an AI astray.

Original question: {question}
Original answer: {answer}

Rules:
- Add a plausible but misleading context sentence before the question
- The misleading context should suggest a wrong answer
- Keep the core question the same

Respond ONLY with valid JSON, no extra text:
{{
    "adversarial_question": "misleading context + original question",
    "misleading_element": "what makes it misleading",
    "correct_response": "what a good AI should say"
}}""",

    "negation_trap": """You are an expert at creating tricky questions that test AI robustness.

Take this question and rewrite it using NEGATION or double negation to confuse AI models.

Original question: {question}
Original answer: {answer}

Rules:
- Use negation words (not, never, except, unless) to flip or complicate the question
- The question should still have a clear correct answer
- Make it grammatically correct but cognitively tricky

Respond ONLY with valid JSON, no extra text:
{{
    "adversarial_question": "your negation-based question",
    "trick_element": "what makes it tricky",
    "correct_response": "what a good AI should say"
}}"""
}


class AdversarialGenerator:
    """
    Uses an LLM to generate adversarial test cases from seed questions.
    Three strategies: false premise, misleading context, negation trap.
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.strategies = list(ADVERSARIAL_PROMPTS.keys())

    def _generate_single(
        self,
        seed_case: TestCase,
        strategy: str
    ) -> TestCase | None:
        """Generate one adversarial case from a seed using a specific strategy."""

        prompt = ADVERSARIAL_PROMPTS[strategy].format(
            question=seed_case.question[:300],  # truncate long questions
            answer=seed_case.reference_answer[:200]
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # some creativity for variety
                max_tokens=512
            )

            raw = response.choices[0].message.content.strip()

            # Clean up common JSON issues
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(raw)
            adversarial_question = data.get("adversarial_question", "")

            if not adversarial_question:
                return None

            return TestCase(
                id=f"adv_{strategy[:4]}_{str(uuid.uuid4())[:8]}",
                question=adversarial_question,
                reference_answer=data.get("correct_response", seed_case.reference_answer),
                category=Category.ADVERSARIAL,
                difficulty=Difficulty.HARD,
                source=f"adversarial_{strategy}",
                context=seed_case.question,  # store original question as context
                tags=["adversarial", strategy, seed_case.source]
            )

        except (json.JSONDecodeError, KeyError, Exception):
            return None  # skip failed generations silently

    def generate(
        self,
        seed_cases: List[TestCase],
        num_adversarial: int = 30,
    ) -> List[TestCase]:
        """
        Generate adversarial cases from seed dataset.
        Cycles through strategies for variety.
        """
        print(f"\n⚔️  Generating {num_adversarial} adversarial test cases...")
        print(f"   Strategies: {', '.join(self.strategies)}")
        print(f"   Model: {self.model}\n")

        adversarial_cases = []
        failed = 0

        with tqdm(total=num_adversarial, desc="Generating") as pbar:
            idx = 0
            attempts = 0
            max_attempts = num_adversarial * 3  # allow retries

            while len(adversarial_cases) < num_adversarial and attempts < max_attempts:
                seed = seed_cases[idx % len(seed_cases)]
                strategy = self.strategies[idx % len(self.strategies)]

                result = self._generate_single(seed, strategy)

                if result:
                    adversarial_cases.append(result)
                    pbar.update(1)
                else:
                    failed += 1

                idx += 1
                attempts += 1

        print(f"\n✅ Generated {len(adversarial_cases)} adversarial cases")
        if failed > 0:
            print(f"   ⚠️  {failed} generations failed/skipped (normal)")

        return adversarial_cases