import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import uuid
import os

# Check for API keys — show friendly error on HF Spaces if missing
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "⚠️ GROQ_API_KEY not found. "
        "Please set it as a Space secret in Settings → Variables and secrets."
    )
    st.stop()

from src.utils.config import load_config
from src.utils.dataset_loader import load_dataset_from_file
from src.utils.regression_store import (
    load_all_runs, load_report, save_report,
    regression_diff, get_latest_run_id, init_db
)
from src.utils.aggregator import aggregate_results
from src.pipelines.groq_pipeline import GroqPipeline
from src.evals.llm_judge import LLMJudge
from src.evals.eval_runner import EvalRunner
from src.evals.reference_metrics import ReferenceMetricsEval

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Benchmark",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .regression-alert {
        background: #3d1a1a;
        border: 1px solid #f38ba8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        color: #f38ba8;
    }
    .improvement-alert {
        background: #1a2e1a;
        border: 1px solid #a6e3a1;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        color: #a6e3a1;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────
def get_delta_str(current, baseline, higher_is_better=True):
    if baseline is None or baseline == 0:
        return ""
    delta = current - baseline
    pct = delta / baseline * 100
    if higher_is_better:
        return f"{'▲' if delta >= 0 else '▼'} {abs(pct):.1f}% vs last run"
    else:
        return f"{'▼' if delta >= 0 else '▲'} {abs(pct):.1f}% vs last run"


def get_baseline_metric(runs, metric, current_run_id):
    for run in runs:
        if run["run_id"] != current_run_id:
            return run.get(metric)
    return None


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 LLM Eval")
    st.markdown("---")
    st.subheader("⚙️ Run Configuration")

    pipeline_model = st.selectbox(
        "Pipeline model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0
    )

    num_cases = st.slider(
        "Test cases",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    dataset_type = st.selectbox(
        "Dataset",
        ["Seed dataset (SQuAD + OpenBookQA)", "Adversarial dataset"],
        index=0
    )

    st.markdown("---")
    run_button = st.button("🚀 Run Eval", type="primary")

    st.markdown("---")
    st.subheader("📂 Past Runs")
    runs = load_all_runs()
    if runs:
        run_options = {
            r["run_id"]: f"{r['run_id']} — {r['judge_pass_rate']}% pass"
            for r in runs
        }
        selected_run = st.selectbox(
            "View run",
            list(run_options.keys()),
            format_func=lambda x: run_options[x]
        )
    else:
        selected_run = None
        st.info("No runs yet. Click Run Eval!")

    st.markdown("---")
    st.caption("Built with Groq · Streamlit · SQLite")


# ── Main header ───────────────────────────────────────────
st.title("🧪 LLM Eval Benchmark Builder")
st.markdown(
    "Automated evaluation — **LLM-as-judge** · "
    "**Self-consistency** · **Reference metrics** · **Regression detection**"
)
st.markdown("---")


# ── Run eval ──────────────────────────────────────────────
if run_button:
    config = load_config()

    with st.status("Running evaluation...", expanded=True) as status:
        st.write("📂 Loading dataset...")
        if "Adversarial" in dataset_type:
            all_cases = load_dataset_from_file("data/adversarial_dataset.json")
        else:
            all_cases = load_dataset_from_file("data/seed_dataset.json")
        test_cases = all_cases[:num_cases]

        st.write(f"🤖 Running {pipeline_model} on {len(test_cases)} cases...")
        pipeline = GroqPipeline(api_key=config.groq_api_key, model=pipeline_model)
        judge = LLMJudge(api_key=config.groq_api_key)
        runner = EvalRunner(pipeline=pipeline, judge=judge)
        results = runner.run(test_cases=test_cases, verbose=False)

        st.write("📐 Computing reference metrics...")
        metrics_eval = ReferenceMetricsEval()
        results = metrics_eval.score_batch(results, test_cases)

        st.write("📊 Aggregating scores...")
        run_id = f"run-{str(uuid.uuid4())[:8]}"
        report = aggregate_results(
            results=results,
            run_id=run_id,
            pipeline_model=pipeline_model,
            timestamp=datetime.now().isoformat()
        )

        save_report(report)
        st.write(f"💾 Saved as {run_id}")
        status.update(label="✅ Eval complete!", state="complete")

    st.session_state["latest_report"] = report
    st.session_state["latest_results"] = results
    st.session_state["latest_cases"] = test_cases
    st.rerun()


# ── Load data to display ──────────────────────────────────
report_data = None
results_data = []

if "latest_report" in st.session_state:
    report_data = st.session_state["latest_report"]
    results_data = st.session_state.get("latest_results", [])
elif selected_run:
    report_data = load_report(selected_run)

if report_data is None:
    st.info("👈 Configure and click **Run Eval** to get started, or select a past run from the sidebar.")
    st.stop()


# ── Normalize report data ─────────────────────────────────
if hasattr(report_data, 'to_dict'):
    rd = report_data.to_dict()
    rd["run_id"] = report_data.run_id
    current_report_obj = report_data
else:
    rd = report_data
    current_report_obj = None

runs = load_all_runs()
current_run_id = rd.get("run_id", "")


# ── Regression / improvement alert ───────────────────────
if current_report_obj and len(runs) >= 2:
    baseline_id = get_latest_run_id()
    if baseline_id and baseline_id != current_run_id:
        diff = regression_diff(current_report_obj, baseline_id)
        if diff.get("has_regressions"):
            reg_text = " · ".join(
                f"{r['metric']} {r['delta_pct']}%"
                for r in diff["regressions"]
            )
            st.markdown(
                f'<div class="regression-alert">🚨 <b>Regressions detected</b> '
                f'vs <code>{baseline_id}</code>: {reg_text}</div>',
                unsafe_allow_html=True
            )
        if diff.get("improvements"):
            imp_text = " · ".join(
                f"{i['metric']} +{i['delta_pct']}%"
                for i in diff["improvements"]
            )
            st.markdown(
                f'<div class="improvement-alert">📈 <b>Improvements</b> '
                f'vs <code>{baseline_id}</code>: {imp_text}</div>',
                unsafe_allow_html=True
            )


# ── Report header ─────────────────────────────────────────
st.subheader(f"📊 {current_run_id}")
st.caption(
    f"Pipeline: `{rd.get('pipeline_model')}` · "
    f"Cases: {rd.get('total_cases')} · "
    f"{str(rd.get('timestamp', ''))[:19]}"
)


# ── Top metric cards ──────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

baseline_pass = get_baseline_metric(runs, "judge_pass_rate", current_run_id)
baseline_score = get_baseline_metric(runs, "judge_avg_score", current_run_id)
baseline_bert = get_baseline_metric(runs, "avg_bert_f1", current_run_id)

with col1:
    val = rd.get("judge_pass_rate", 0)
    delta = get_delta_str(val, baseline_pass)
    st.metric("✅ Pass Rate", f"{val}%", delta)

with col2:
    val = rd.get("judge_avg_score", 0)
    delta = get_delta_str(val, baseline_score)
    st.metric("🏆 Avg Judge Score", f"{val}/100", delta)

with col3:
    val = rd.get("avg_bert_f1", 0)
    delta = get_delta_str(val, baseline_bert)
    st.metric("🧠 BERTScore F1", f"{val:.3f}", delta)

with col4:
    cost = rd.get("total_cost_usd", 0)
    latency = rd.get("avg_latency_seconds", 0)
    st.metric("⚡ Avg Latency", f"{latency}s", f"${cost:.6f} total cost")


st.markdown("---")


# ── Eval score bars ───────────────────────────────────────
st.subheader("📐 Eval Scores Breakdown")

eval_metrics = {
    "Judge Score /100": rd.get("judge_avg_score", 0),
    "Correctness /5×20": rd.get("judge_avg_correctness", 0) * 20,
    "Relevance /5×20": rd.get("judge_avg_relevance", 0) * 20,
    "Completeness /5×20": rd.get("judge_avg_completeness", 0) * 20,
    "Hallucination /5×20": rd.get("judge_avg_hallucination", 0) * 20,
    "ROUGE-L ×100": rd.get("avg_rouge_l", 0) * 100,
    "BERTScore F1 ×100": rd.get("avg_bert_f1", 0) * 100,
    "Exact Match ×100": rd.get("avg_exact_match", 0) * 100,
}

colors = [
    "#89b4fa" if v >= 70 else "#fab387" if v >= 40 else "#f38ba8"
    for v in eval_metrics.values()
]

fig_bars = go.Figure(go.Bar(
    x=list(eval_metrics.values()),
    y=list(eval_metrics.keys()),
    orientation='h',
    marker_color=colors,
    text=[f"{v:.1f}" for v in eval_metrics.values()],
    textposition='outside'
))
fig_bars.update_layout(
    xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#313244"),
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=320,
    margin=dict(l=10, r=40, t=10, b=10),
    font=dict(color="#cdd6f4")
)
st.plotly_chart(fig_bars, use_container_width=True)


# ── Run history ───────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Run History")

if runs:
    df_runs = pd.DataFrame(runs)
    df_runs = df_runs[[
        "run_id", "pipeline_model", "total_cases",
        "judge_pass_rate", "judge_avg_score",
        "avg_bert_f1", "avg_rouge_l", "total_cost_usd"
    ]].rename(columns={
        "run_id": "Run ID",
        "pipeline_model": "Model",
        "total_cases": "Cases",
        "judge_pass_rate": "Pass Rate %",
        "judge_avg_score": "Judge Score",
        "avg_bert_f1": "BERTScore",
        "avg_rouge_l": "ROUGE-L",
        "total_cost_usd": "Cost ($)"
    })

    # Color pass rate column
    st.dataframe(
        df_runs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pass Rate %": st.column_config.ProgressColumn(
                "Pass Rate %",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "Judge Score": st.column_config.ProgressColumn(
                "Judge Score",
                min_value=0,
                max_value=100,
                format="%.1f"
            )
        }
    )
else:
    st.info("No past runs yet.")


# ── Category breakdown ────────────────────────────────────
st.markdown("---")
cat_scores = rd.get("category_scores", {})
if cat_scores:
    st.subheader("🗂️ Score by Category")
    cat_cols = st.columns(len(cat_scores))
    for i, (cat, score) in enumerate(cat_scores.items()):
        with cat_cols[i]:
            st.metric(cat.capitalize(), f"{score}/100")


# ── Failure analysis ──────────────────────────────────────
st.markdown("---")
st.subheader("❌ Failure Analysis")

if results_data:
    failures = [
        r for r in results_data
        if r.metadata.get("verdict") == "FAIL"
    ]

    if failures:
        st.markdown(f"**{len(failures)} failed cases** out of {len(results_data)}")
        for result in failures[:5]:
            judge = result.scores.get("judge", {})
            with st.expander(
                f"❌ {result.test_case_id} — "
                f"Score: {judge.get('overall_score', 0)}/100"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Pipeline output:**")
                    st.markdown(result.pipeline_output[:300])
                with col_b:
                    st.markdown("**Judge reasoning:**")
                    st.markdown(judge.get("reasoning", "N/A"))
                    st.markdown(f"Correctness: `{judge.get('correctness')}/5`")
                    st.markdown(f"Relevance: `{judge.get('relevance')}/5`")
                    st.markdown(f"Hallucination: `{judge.get('hallucination')}/5`")
    else:
        st.success("🎉 No failures in this run!")
else:
    st.info("Run an eval to see failure analysis.")

# ── Cost vs Quality Bubble Chart ──────────────────────────
st.markdown("---")
st.subheader("💰 Cost vs Quality — Model Comparison")

if runs and len(runs) >= 2:
    bubble_data = []
    for run in runs:
        bubble_data.append({
            "Model": run["pipeline_model"],
            "Run ID": run["run_id"],
            "Quality Score": run["judge_avg_score"],
            "Cost per run ($)": run["total_cost_usd"],
            "Latency (s)": run["avg_latency_seconds"],
            "Pass Rate": run["judge_pass_rate"],
            "Cases": run["total_cases"]
        })

    df_bubble = pd.DataFrame(bubble_data)

    fig_bubble = px.scatter(
        df_bubble,
        x="Cost per run ($)",
        y="Quality Score",
        size="Latency (s)",
        color="Model",
        hover_name="Run ID",
        hover_data={
            "Pass Rate": True,
            "Cases": True,
            "Latency (s)": True,
            "Cost per run ($)": ":.6f",
            "Quality Score": ":.1f"
        },
        size_max=50,
        title="Cost vs Quality — bubble size = latency"
    )

    fig_bubble.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd6f4"),
        height=420,
        xaxis=dict(showgrid=True, gridcolor="#313244"),
        yaxis=dict(showgrid=True, gridcolor="#313244", range=[0, 105]),
    )

    fig_bubble.add_hline(
        y=60,
        line_dash="dash",
        line_color="#f38ba8",
        annotation_text="Pass threshold (60)",
        annotation_position="bottom right"
    )

    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption("Each bubble = one eval run. Bigger bubble = slower response. Ideal = top-left corner (high quality, low cost).")

elif runs and len(runs) == 1:
    st.info("Run eval at least twice to see the cost vs quality comparison.")
else:
    st.info("No runs yet. Click Run Eval to get started.")


# ── Judge dimension radar chart ───────────────────────────
st.markdown("---")
st.subheader("🕸️ Judge Dimensions Radar")

if report_data:
    categories = [
        "Correctness", "Relevance",
        "Completeness", "Hallucination"
    ]
    values = [
        rd.get("judge_avg_correctness", 0),
        rd.get("judge_avg_relevance", 0),
        rd.get("judge_avg_completeness", 0),
        rd.get("judge_avg_hallucination", 0),
    ]
    # Close the radar loop
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(137, 180, 250, 0.2)',
        line=dict(color='#89b4fa', width=2),
        name=rd.get("pipeline_model", "model")
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                showgrid=True,
                gridcolor="#313244",
                color="#6c7086"
            ),
            angularaxis=dict(color="#cdd6f4"),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdd6f4"),
        height=380,
        showlegend=True,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Radar shows avg score per judge dimension (0–5). Bigger area = better overall quality.")

# ── Adversarial vs Normal comparison ─────────────────────
st.markdown("---")
st.subheader("⚔️ Adversarial vs Normal Performance")

if results_data:
    normal = [
        r for r in results_data
        if "adversarial" not in r.metadata.get("source", "")
    ]
    adversarial = [
        r for r in results_data
        if "adversarial" in r.metadata.get("source", "")
    ]

    def avg_score(result_list):
        scores = [
            r.scores.get("judge", {}).get("overall_score", 0)
            for r in result_list
        ]
        return round(sum(scores) / len(scores), 1) if scores else 0

    def pass_rate(result_list):
        if not result_list:
            return 0
        passed = sum(
            1 for r in result_list
            if r.metadata.get("verdict") == "PASS"
        )
        return round(passed / len(result_list) * 100, 1)

    col_n, col_a = st.columns(2)

    with col_n:
        st.markdown("**📝 Normal cases**")
        if normal:
            st.metric("Cases", len(normal))
            st.metric("Pass rate", f"{pass_rate(normal)}%")
            st.metric("Avg judge score", f"{avg_score(normal)}/100")
        else:
            st.info("No normal cases in this run.")

    with col_a:
        st.markdown("**⚔️ Adversarial cases**")
        if adversarial:
            st.metric("Cases", len(adversarial))
            st.metric("Pass rate", f"{pass_rate(adversarial)}%")
            st.metric("Avg judge score", f"{avg_score(adversarial)}/100")
        else:
            st.info("Run with adversarial dataset to see comparison.")

    if normal and adversarial:
        drop = avg_score(normal) - avg_score(adversarial)
        if drop > 10:
            st.warning(
                f"⚠️ Model scores **{drop:.1f} points lower** on adversarial "
                f"cases — it's vulnerable to tricky inputs."
            )
        else:
            st.success(
                f"✅ Model is robust — only **{drop:.1f} point drop** "
                f"on adversarial cases."
            )
else:
    st.info("Run an eval to see adversarial vs normal breakdown.")


# ── Export report ─────────────────────────────────────────
st.markdown("---")
st.subheader("📤 Export Report")

if report_data:
    import json

    if hasattr(report_data, 'to_dict'):
        export_dict = report_data.to_dict()
    else:
        export_dict = report_data

    export_json = json.dumps(export_dict, indent=2)

    col_exp1, col_exp2 = st.columns([1, 3])
    with col_exp1:
        st.download_button(
            label="⬇️ Download JSON report",
            data=export_json,
            file_name=f"{rd.get('run_id', 'report')}.json",
            mime="application/json"
        )
    with col_exp2:
        st.caption(
            f"Downloads full eval report for `{rd.get('run_id')}` "
            f"including all scores, metadata, and category breakdown."
        )


# ── Summary insight ───────────────────────────────────────
st.markdown("---")
st.subheader("💡 Auto Insight")

if report_data:
    insights = []

    pass_rate_val = rd.get("judge_pass_rate", 0)
    bert_val = rd.get("avg_bert_f1", 0)
    hallucination_val = rd.get("judge_avg_hallucination", 0)
    exact_val = rd.get("avg_exact_match", 0)
    rouge_val = rd.get("avg_rouge_l", 0)

    if pass_rate_val >= 80:
        insights.append("✅ **Strong pipeline** — pass rate above 80%. Ready for production testing.")
    elif pass_rate_val >= 60:
        insights.append("⚠️ **Decent pipeline** — pass rate between 60-80%. Room for prompt improvement.")
    else:
        insights.append("🚨 **Weak pipeline** — pass rate below 60%. Significant prompt engineering needed.")

    if hallucination_val < 3.0:
        insights.append("🚨 **High hallucination risk** — avg hallucination score below 3/5. Check failure cases.")
    else:
        insights.append("✅ **Low hallucination** — model stays grounded in most responses.")

    if bert_val >= 0.75:
        insights.append("✅ **High semantic accuracy** — BERTScore above 0.75 means answers are meaningfully correct.")
    elif bert_val >= 0.60:
        insights.append("⚠️ **Moderate semantic accuracy** — BERTScore between 0.60-0.75.")
    else:
        insights.append("🚨 **Low semantic accuracy** — BERTScore below 0.60. Answers are drifting from reference.")

    if exact_val < 0.1:
        insights.append("ℹ️ **Low exact match** — model gives verbose answers. Consider prompting for concise responses.")

    for insight in insights:
        st.markdown(insight)


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c7086; font-size: 0.8rem; padding: 1rem 0;'>
        🧪 LLM Eval Benchmark Builder · Built by
        <a href='https://github.com/Anand-Tiwari2404' style='color: #89b4fa;'>Anand Tiwari</a>
        · Powered by Groq · Streamlit · SQLite
    </div>
    """,
    unsafe_allow_html=True
)