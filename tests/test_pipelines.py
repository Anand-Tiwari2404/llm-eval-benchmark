import pytest
from src.utils.config import load_config
from src.pipelines import GroqPipeline, GeminiPipeline, PipelineResponse


@pytest.fixture
def config():
    return load_config()


def test_groq_pipeline_returns_correct_schema(config):
    pipeline = GroqPipeline(api_key=config.groq_api_key)
    response = pipeline.run("What is 2 + 2? Answer in one word.")

    assert isinstance(response, PipelineResponse)
    assert len(response.output) > 0
    assert response.model == "llama-3.1-8b-instant"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_seconds > 0
    assert response.cost_usd >= 0
    print(f"\n✅ Groq response: {response.output}")
    print(f"   Tokens: {response.input_tokens} in / {response.output_tokens} out")
    print(f"   Latency: {response.latency_seconds}s | Cost: ${response.cost_usd}")


def test_gemini_pipeline_returns_correct_schema(config):
    pipeline = GeminiPipeline(api_key=config.gemini_api_key)
    response = pipeline.run("What is 2 + 2? Answer in one word.")

    assert isinstance(response, PipelineResponse)
    assert len(response.output) > 0
    assert response.latency_seconds > 0
    print(f"\n✅ Gemini response: {response.output}")
    print(f"   Tokens: {response.input_tokens} in / {response.output_tokens} out")
    print(f"   Latency: {response.latency_seconds}s | Cost: ${response.cost_usd}")


def test_pipeline_handles_errors_gracefully(config):
    # Test with a bad model name — should return error response, not crash
    pipeline = GroqPipeline(api_key=config.groq_api_key, model="fake-model-xyz")
    response = pipeline.run("Hello")

    assert isinstance(response, PipelineResponse)
    assert "ERROR" in response.output
    assert response.metadata["error"] is True
    print(f"\n✅ Error handled gracefully: {response.output[:80]}")


def test_both_pipelines_answer_same_question(config):
    question = "What is the capital of France? Answer in one word."

    groq = GroqPipeline(api_key=config.groq_api_key)
    gemini = GeminiPipeline(api_key=config.gemini_api_key)

    groq_response = groq.run(question)
    gemini_response = gemini.run(question)

    print(f"\n✅ Groq says: {groq_response.output.strip()}")
    print(f"✅ Gemini says: {gemini_response.output.strip()}")
    print(f"\n📊 Latency comparison:")
    print(f"   Groq:   {groq_response.latency_seconds}s")
    print(f"   Gemini: {gemini_response.latency_seconds}s")

    assert "paris" in groq_response.output.lower()
    assert "paris" in gemini_response.output.lower()