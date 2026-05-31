# LLM Eval Benchmark Builder

An automated evaluation framework for LLM pipelines — generates test cases, runs multi-dimensional evals (LLM-as-judge, self-consistency, reference metrics), detects regressions, and visualizes results on a live dashboard.

## Features
- Pluggable pipeline interface (Groq, Gemini, any LLM)
- Auto-generates normal + adversarial test cases
- LLM-as-judge with structured rubric scoring
- Self-consistency hallucination detection
- Regression tracking across runs
- Streamlit dashboard with cost vs quality analysis

## Stack
Python · LangChain · Groq · Gemini · Streamlit · SQLite · Plotly

## Live Demo
_Coming soon_

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Add your API keys to `.env`, then:
```bash
python -m pytest tests/
```