import json
import sqlite3
import os
from datetime import datetime
from typing import List, Optional
from src.utils.aggregator import EvalReport


DB_PATH = "outputs/eval_runs.db"

REGRESSION_THRESHOLD = 0.05  # flag if metric drops more than 5%


def init_db(db_path: str = DB_PATH) -> None:
    """Create the SQLite database and tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            pipeline_model TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_cases INTEGER,
            judge_pass_rate REAL,
            judge_avg_score REAL,
            avg_rouge_l REAL,
            avg_bert_f1 REAL,
            avg_exact_match REAL,
            consistency_avg REAL,
            total_cost_usd REAL,
            avg_latency_seconds REAL,
            full_report TEXT  -- JSON blob of complete report
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {db_path}")


def save_report(report: EvalReport, db_path: str = DB_PATH) -> None:
    """Save an EvalReport to the database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Serialize report without raw results (too large)
    report_dict = report.to_dict()

    cursor.execute("""
        INSERT OR REPLACE INTO eval_runs (
            run_id, pipeline_model, timestamp, total_cases,
            judge_pass_rate, judge_avg_score, avg_rouge_l,
            avg_bert_f1, avg_exact_match, consistency_avg,
            total_cost_usd, avg_latency_seconds, full_report
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report.run_id,
        report.pipeline_model,
        report.timestamp,
        report.total_cases,
        report.judge_pass_rate,
        report.judge_avg_score,
        report.avg_rouge_l,
        report.avg_bert_f1,
        report.avg_exact_match,
        report.consistency_avg,
        report.total_cost_usd,
        report.avg_latency_seconds,
        json.dumps(report_dict)
    ))

    conn.commit()
    conn.close()
    print(f"💾 Saved run '{report.run_id}' to database")


def load_all_runs(db_path: str = DB_PATH) -> List[dict]:
    """Load all eval runs from the database."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, pipeline_model, timestamp, total_cases,
               judge_pass_rate, judge_avg_score, avg_rouge_l,
               avg_bert_f1, avg_exact_match, consistency_avg,
               total_cost_usd, avg_latency_seconds
        FROM eval_runs
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    columns = [
        "run_id", "pipeline_model", "timestamp", "total_cases",
        "judge_pass_rate", "judge_avg_score", "avg_rouge_l",
        "avg_bert_f1", "avg_exact_match", "consistency_avg",
        "total_cost_usd", "avg_latency_seconds"
    ]
    return [dict(zip(columns, row)) for row in rows]


def load_report(run_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Load a specific run's full report."""
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT full_report FROM eval_runs WHERE run_id = ?",
        (run_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return json.loads(row[0])
    return None


def regression_diff(
    current: EvalReport,
    baseline_run_id: str,
    db_path: str = DB_PATH
) -> dict:
    """
    Compare current run against a baseline run.
    Flags any metric that dropped more than REGRESSION_THRESHOLD.
    """
    baseline = load_report(baseline_run_id, db_path)
    if not baseline:
        return {"error": f"Baseline run '{baseline_run_id}' not found"}

    metrics_to_compare = [
        ("judge_pass_rate", "Judge pass rate", "%"),
        ("judge_avg_score", "Judge avg score", "/100"),
        ("avg_rouge_l", "ROUGE-L", ""),
        ("avg_bert_f1", "BERTScore F1", ""),
        ("avg_exact_match", "Exact match", ""),
        ("consistency_avg", "Consistency avg", ""),
    ]

    regressions = []
    improvements = []
    unchanged = []

    for metric_key, metric_name, unit in metrics_to_compare:
        current_val = getattr(current, metric_key, 0)
        baseline_val = baseline.get(metric_key, 0)

        if baseline_val == 0:
            continue

        delta = current_val - baseline_val
        delta_pct = delta / baseline_val

        entry = {
            "metric": metric_name,
            "current": round(current_val, 4),
            "baseline": round(baseline_val, 4),
            "delta": round(delta, 4),
            "delta_pct": round(delta_pct * 100, 2)
        }

        if delta_pct < -REGRESSION_THRESHOLD:
            regressions.append(entry)
        elif delta_pct > REGRESSION_THRESHOLD:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    return {
        "current_run": current.run_id,
        "baseline_run": baseline_run_id,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "has_regressions": len(regressions) > 0
    }


def get_latest_run_id(db_path: str = DB_PATH) -> Optional[str]:
    """Get the most recent run ID from the database."""
    runs = load_all_runs(db_path)
    if len(runs) >= 2:
        return runs[1]["run_id"]  # second most recent = baseline
    return None