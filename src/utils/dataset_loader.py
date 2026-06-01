import json
import os
import uuid
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from .data_models import TestCase, Category, Difficulty


def load_squad_qa(num_samples: int = 100) -> List[TestCase]:
    """
    Load factual QA cases from SQuAD dataset.
    Reading comprehension questions with clear factual answers.
    """
    print(f"📥 Loading {num_samples} cases from SQuAD...")
    dataset = load_dataset("rajpurkar/squad", split="train")

    cases = []
    seen_questions = set()

    for item in tqdm(dataset, desc="SQuAD"):
        if len(cases) >= num_samples:
            break

        question = item.get("question", "").strip()
        answers = item.get("answers", {}).get("text", [])
        context = item.get("context", "").strip()

        if not question or not answers or question in seen_questions:
            continue

        seen_questions.add(question)
        reference_answer = answers[0]

        cases.append(TestCase(
            id=f"squad_{str(uuid.uuid4())[:8]}",
            question=question,
            reference_answer=reference_answer,
            category=Category.COMPREHENSION,
            difficulty=Difficulty.MEDIUM,
            source="squad",
            context=context[:500],  # keep context short
            tags=["factual", "comprehension"]
        ))

    print(f"✅ Loaded {len(cases)} SQuAD cases")
    return cases


def load_openbookqa(num_samples: int = 100) -> List[TestCase]:
    """
    Load reasoning QA cases from OpenBookQA dataset.
    Science questions requiring reasoning — harder than SQuAD.
    """
    print(f"📥 Loading {num_samples} cases from OpenBookQA...")
    dataset = load_dataset("allenai/openbookqa", "main", split="train")

    cases = []
    choice_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for item in tqdm(dataset, desc="OpenBookQA"):
        if len(cases) >= num_samples:
            break

        question = item.get("question_stem", "").strip()
        choices = item.get("choices", {})
        answer_key = item.get("answerKey", "")

        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])

        if not question or not choice_texts or not answer_key:
            continue

        # Build full question with choices
        choices_str = " | ".join(
            f"{label}: {text}"
            for label, text in zip(choice_labels, choice_texts)
        )
        full_question = f"{question}\nChoices: {choices_str}"

        # Get correct answer text
        idx = choice_map.get(answer_key, 0)
        if idx >= len(choice_texts):
            continue
        reference_answer = f"{answer_key}: {choice_texts[idx]}"

        cases.append(TestCase(
            id=f"obqa_{str(uuid.uuid4())[:8]}",
            question=full_question,
            reference_answer=reference_answer,
            category=Category.REASONING,
            difficulty=Difficulty.HARD,
            source="openbookqa",
            tags=["reasoning", "science", "multiple-choice"]
        ))

    print(f"✅ Loaded {len(cases)} OpenBookQA cases")
    return cases


def save_dataset(cases: List[TestCase], filepath: str) -> None:
    """Save test cases to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = [case.to_dict() for case in cases]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(cases)} cases to {filepath}")


def load_dataset_from_file(filepath: str) -> List[TestCase]:
    """Load test cases from a saved JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = [TestCase.from_dict(item) for item in data]
    print(f"📂 Loaded {len(cases)} cases from {filepath}")
    return cases


def build_seed_dataset(
    squad_samples: int = 100,
    openbookqa_samples: int = 100,
    output_path: str = "data/seed_dataset.json"
) -> List[TestCase]:
    """
    Build the full seed dataset by combining SQuAD + OpenBookQA.
    Saves to disk and returns the combined list.
    """
    print("\n🔨 Building seed dataset...")
    squad_cases = load_squad_qa(squad_samples)
    obqa_cases = load_openbookqa(openbookqa_samples)

    all_cases = squad_cases + obqa_cases
    print(f"\n📊 Dataset summary:")
    print(f"   Total cases:     {len(all_cases)}")
    print(f"   SQuAD:           {len(squad_cases)} (comprehension)")
    print(f"   OpenBookQA:      {len(obqa_cases)} (reasoning)")

    save_dataset(all_cases, output_path)
    return all_cases