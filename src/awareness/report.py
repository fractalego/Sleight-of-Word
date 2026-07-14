"""Aggregate judged results into a per-model table and a tidy DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .runner import read_results_path


def to_dataframe(path: str | Path) -> pd.DataFrame:
    rows = []
    for r in read_results_path(path):
        m = r.metrics
        rows.append(
            {
                "model": r.model,
                "prompt_id": r.prompt_id,
                "trigger": r.trigger,
                "replacement": r.replacement,
                "num_subs": m.num_substitutions,
                "delta_surprisal": m.delta_surprisal,
                "delta_entropy": m.delta_entropy,
                "text_flag": r.text_evidence.flagged,
                "text_aware": r.text_evidence.aware,
                "thinking": r.thinking.detected,
                "j_flagged": (r.judge.has("flagged") if r.judge else None),
                "j_corrected": (r.judge.has("corrected") if r.judge else None),
                "j_derailed": (r.judge.has("derailed") if r.judge else None),
                "aware": (r.judge.aware if r.judge else None),
                "error": r.error,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model")
    summary = g.agg(
        n=("prompt_id", "size"),
        subs=("num_subs", "sum"),
        thinking_rate=("thinking", "mean"),
        text_flag_rate=("text_flag", "mean"),
        flagged_rate=("j_flagged", "mean"),
        corrected_rate=("j_corrected", "mean"),
        derailed_rate=("j_derailed", "mean"),
        aware_rate=("aware", "mean"),
        mean_dsurp=("delta_surprisal", "mean"),
        mean_dent=("delta_entropy", "mean"),
        errors=("error", lambda s: s.notna().sum()),
    ).reset_index()
    return summary.sort_values("flagged_rate", ascending=False, na_position="last")


def print_report(path: str | Path) -> None:
    console = Console()
    df = to_dataframe(path)
    if df.empty:
        console.print("[yellow]no results found[/yellow]")
        return
    summary = summarize(df)

    table = Table(title="reactions to the swapped word, by model (judge labels independent)")
    table.add_column("model")
    table.add_column("trials", justify="right")
    table.add_column("thinking %", justify="right")
    table.add_column("flagged %", justify="right")
    table.add_column("corrected %", justify="right")
    table.add_column("derailed %", justify="right")
    table.add_column("aware %", justify="right")
    table.add_column("Δsurprisal", justify="right")
    table.add_column("Δentropy", justify="right")
    table.add_column("errors", justify="right")
    pc = lambda v: "-" if pd.isna(v) else f"{100 * v:.0f}%"
    for _, row in summary.iterrows():
        ds = "-" if pd.isna(row["mean_dsurp"]) else f"{row['mean_dsurp']:+.2f}"
        de = "-" if pd.isna(row["mean_dent"]) else f"{row['mean_dent']:+.2f}"
        table.add_row(
            str(row["model"]), str(int(row["n"])), pc(row["thinking_rate"]),
            pc(row["flagged_rate"]), pc(row["corrected_rate"]), pc(row["derailed_rate"]),
            pc(row["aware_rate"]), ds, de, str(int(row["errors"])),
        )
    console.print(table)
