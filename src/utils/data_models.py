from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    COMPREHENSION = "comprehension"
    ADVERSARIAL = "adversarial"


@dataclass
class TestCase:
    """A single test case for evaluating an LLM pipeline."""
    id: str
    question: str
    reference_answer: str
    category: Category
    difficulty: Difficulty
    source: str                          # which dataset it came from
    context: Optional[str] = None        # supporting passage if available
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "source": self.source,
            "context": self.context,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(
            id=data["id"],
            question=data["question"],
            reference_answer=data["reference_answer"],
            category=Category(data["category"]),
            difficulty=Difficulty(data["difficulty"]),
            source=data["source"],
            context=data.get("context"),
            tags=data.get("tags", [])
        )


@dataclass
class EvalResult:
    """Result of running one TestCase through one pipeline."""
    test_case_id: str
    pipeline_model: str
    pipeline_output: str
    latency_seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    scores: dict = field(default_factory=dict)   # filled by eval runners
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "pipeline_model": self.pipeline_model,
            "pipeline_output": self.pipeline_output,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "scores": self.scores,
            "metadata": self.metadata
        }