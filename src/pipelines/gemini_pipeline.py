import time
from google import genai
from .base import BasePipeline, PipelineResponse

# Gemini pricing per 1M tokens
GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


class GeminiPipeline(BasePipeline):
    """
    Pipeline wrapper for Google Gemini models.
    Used primarily as the judge LLM in our eval system.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 temperature: float = 0.0, max_tokens: int = 1024):
        super().__init__(model, temperature, max_tokens)
        self.client = genai.Client(api_key=api_key)

    def _call_api(self, prompt: str) -> PipelineResponse:
        start = time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        latency = time.time() - start

        # Extract token counts
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

        # Calculate cost
        pricing = GEMINI_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        return PipelineResponse(
            output=response.text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=round(latency, 3),
            cost_usd=round(cost, 6),
            metadata={"provider": "gemini"}
        )