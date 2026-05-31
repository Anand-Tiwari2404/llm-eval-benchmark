import pytest
from src.utils.config import load_config
from groq import Groq
from google import genai

def test_config_loads():
    config = load_config()
    assert config.groq_api_key is not None
    assert config.gemini_api_key is not None
    print("\n✅ Config loaded successfully")

def test_groq_call():
    config = load_config()
    client = Groq(api_key=config.groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        max_tokens=20
    )
    reply = response.choices[0].message.content
    print(f"\n✅ Groq response: {reply}")
    assert len(reply) > 0

def test_gemini_call():
    config = load_config()
    client = genai.Client(api_key=config.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in exactly 5 words."
    )
    reply = response.text
    print(f"\n✅ Gemini response: {reply}")
    assert len(reply) > 0