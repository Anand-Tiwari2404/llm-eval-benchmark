---
title: LLM Eval Benchmark Builder
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
app_port: 7860
---

# 🧪 LLM Eval Benchmark Builder

> **Automated evaluation infrastructure for LLM pipelines** — because shipping AI without evals is flying blind.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3-orange?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-35+_passing-brightgreen?style=flat-square)](./tests)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](./LICENSE)

---

## 🎯 The Problem

Every AI team building with LLMs faces the same question: **"Is my pipeline actually working?"**

Eyeballing outputs doesn't scale. Changing a single prompt can silently break factual accuracy, increase hallucinations, or tank relevance — and you'd never know without a systematic eval framework.

This project is that framework.

---

## 🚀 Live Demo

> 🔗 **[Coming soon — deploying to Hugging Face Spaces]**

---

## ✨ What It Does

Point it at any LLM pipeline. It automatically:

1. **Generates test cases** from real QA datasets (SQuAD + OpenBookQA) + adversarial variants with false premises, misleading context, and negation traps
2. **Runs 3 independent eval layers** on every response
3. **Detects regressions** — flags any metric that drops >5% vs the previous run
4. **Visualizes everything** on a live Streamlit dashboard

---

## 🔬 3 Independent Eval Layers

### Layer 1 — LLM-as-Judge
A larger model (`llama-3.3-70b`) scores every response on 4 dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Correctness | 40% | Factual accuracy vs reference |
| Relevance | 25% | Does it actually answer the question? |
| Completeness | 20% | Is key information missing? |
| Hallucination | 15% | Did it make anything up? |

### Layer 2 — Self-Consistency
Runs the same question **5× at high temperature**. Inconsistent answers = the model is guessing = hallucination signal.

### Layer 3 — Reference Metrics
Deterministic, no API calls, runs locally:
- **Exact Match** — strict string comparison
- **Token F1** — word overlap score
- **ROUGE-L** — longest common subsequence
- **BERTScore** — semantic similarity via DistilBERT embeddings

---

## 🏗️ Architecture

```
llm-eval-benchmark/
├── src/
│   ├── pipelines/
│   │   ├── base.py                   # Pluggable pipeline interface
│   │   ├── groq_pipeline.py          # Groq (LLaMA 3) wrapper
│   │   └── gemini_pipeline.py        # Gemini wrapper
│   ├── evals/
│   │   ├── llm_judge.py              # LLM-as-judge with rubric scoring
│   │   ├── self_consistency.py       # Hallucination detection
│   │   ├── reference_metrics.py      # ROUGE + BERTScore + F1
│   │   ├── adversarial_generator.py  # Auto adversarial test cases
│   │   └── eval_runner.py            # Orchestrates full eval pipeline
│   └── utils/
│       ├── aggregator.py             # Score aggregation into EvalReport
│       ├── regression_store.py       # SQLite versioned run history
│       ├── dataset_loader.py         # HuggingFace dataset integration
│       └── data_models.py            # TestCase, EvalResult dataclasses
├── app.py                            # Streamlit dashboard
├── run_full_eval.py                  # CLI eval runner
├── build_dataset.py                  # Dataset builder
└── tests/                            # 35+ passing tests
```

## 📊 Dashboard

The Streamlit dashboard gives you:

- **Top metric cards** — pass rate, avg judge score, BERTScore, latency with delta vs last run
- **Score breakdown bars** — all eval dimensions visualized, color-coded by threshold
- **Run history table** — every run with progress bars for pass rate and judge score
- **Regression alerts** — automatic banner when any metric drops >5%
- **Failure analysis** — click any failed case to see the output, judge reasoning, and per-dimension scores
- **Cost vs quality chart** — bubble chart comparing models by cost, quality, and latency *(coming soon)*

---

## ⚡ Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Anand-Tiwari2404/llm-eval-benchmark.git
cd llm-eval-benchmark
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Add API keys
```bash
# .env
GROQ_API_KEY=your_groq_key_here      # free at console.groq.com
GEMINI_API_KEY=your_gemini_key_here  # free at aistudio.google.com
```

### 3. Build the dataset
```bash
python build_dataset.py
python generate_adversarial.py
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

### 5. Or run from CLI
```bash
python run_full_eval.py
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

| Test File | Status | What it tests |
|---|---|---|
| `test_setup.py` | ✅ 3 passed | API connectivity — Groq + Gemini |
| `test_pipelines.py` | ✅ 4 passed | Pipeline wrappers + error handling |
| `test_dataset.py` | ✅ 5 passed | Dataset loading + structure validation |
| `test_adversarial.py` | ✅ 5 passed | Adversarial test case generation |
| `test_llm_judge.py` | ✅ 4 passed | LLM judge scoring + all dimensions |
| `test_self_consistency.py` | ✅ 4 passed | Consistency detection + batch scoring |
| `test_reference_metrics.py` | ✅ 6 passed | ROUGE-L + BERTScore + Exact Match |
| `test_aggregator.py` | ✅ 4 passed | Regression detection + SQLite store |
| **Total** | **✅ 35 passed** | **0 failed** |

## 🗺️ Roadmap

- [x] Pluggable pipeline interface
- [x] SQuAD + OpenBookQA seed dataset (200 cases)
- [x] Adversarial test generator (3 strategies, 30 cases)
- [x] LLM-as-judge with 4-dimension rubric
- [x] Self-consistency hallucination detector
- [x] ROUGE-L + BERTScore + Exact Match + F1
- [x] Score aggregator + SQLite regression store
- [x] Streamlit dashboard with run history
- [x] Cost vs quality bubble chart
- [ ] Multi-model comparison (Groq vs Gemini)
- [ ] Deploy to Hugging Face Spaces
- [ ] Docker container

---

## 💡 Key Design Decisions

**Why 3 eval layers?** Each catches different failure modes. LLM-as-judge catches reasoning errors. Self-consistency catches uncertainty. Reference metrics catch word-level accuracy. A model can fool one layer but rarely all three.

**Why Groq?** Speed + free tier. `llama-3.1-8b-instant` responds in ~0.23s, making it practical to run 200 eval cases without burning through budget.

**Why a larger model as judge?** The judge (`llama-3.3-70b`) needs to reason about quality, not just pattern-match. Using a bigger model as judge and smaller as pipeline mirrors how production eval systems are built at AI companies.

**Why SQLite?** This is a developer tool, not a web app. SQLite gives you versioned run history, regression diffs, and zero infrastructure overhead.

---

## 🤝 Built By

**Anand Tiwari** 

> *"Most AI projects call an API and call it done. This one asks: how do you know it's actually working?"*

---

⭐ Star this repo if you find it useful!