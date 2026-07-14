"""``awareness`` CLI: generate -> judge -> report, with per-run archival.

    uv run awareness generate --only qwen2.5-14b          # -> results/runs/<run>/
    uv run awareness judge    --in results/runs/<run>     # judges every model in the run
    uv run awareness report   --in results/runs/<run>     # one run, or pass results/runs for all
    uv run awareness rescore  --in results/runs/<run>     # re-apply the text detector

Each ``generate`` creates a timestamped run dir holding one ``<model>.raw.jsonl`` per
model (model-based archival names) plus a ``manifest.json``. ``judge`` writes
``<model>.judged.jsonl`` alongside, and every run appends summary rows to
``results/runs/index.csv`` so nothing is ever overwritten.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from .config import load_experiment, load_models

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

RUNS_ROOT = Path("results/runs")
INDEX_CSV = RUNS_ROOT / "index.csv"


@app.command()
def generate(
    models: Path = typer.Option("configs/models.yaml"),
    experiment: Path = typer.Option("configs/experiment.yaml"),
    only: str = typer.Option("", help="comma-separated model names to run (default: all)"),
    run_name: str = typer.Option("", help="run dir name (default: UTC timestamp)"),
    overwrite: bool = typer.Option(False, help="re-run models whose output already exists"),
):
    """Phase 1: chat models answer; swap the trigger word in their reply; measure."""
    from .runner import _done_keys, run_model  # lazy (pulls vLLM)

    specs = load_models(models)
    exp = load_experiment(experiment)
    if only:
        wanted = {s.strip() for s in only.split(",")}
        specs = [s for s in specs if s.name in wanted]

    run_id = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_trials = len(exp.prompts) * len(exp.pairs)
    console.print(
        f"run [bold]{run_id}[/bold] -> {run_dir}\n"
        f"[bold]{n_trials}[/bold] trials/model x [bold]{len(specs)}[/bold] models"
    )

    written = []
    for spec in specs:
        console.rule(f"[cyan]{spec.name}[/cyan] ({spec.model})")
        out = run_dir / f"{spec.name}.raw.jsonl"
        # run_model checkpoints per trial and resumes from out (skips done trials unless
        # --overwrite); a fully-complete model returns instantly without loading weights.
        have = 0 if overwrite else len(_done_keys(out))
        if have >= n_trials and not overwrite:
            console.print(f"  [yellow]skip[/yellow] {spec.name}: {have}/{n_trials} already done")
            written.append({"model": spec.name, "hf": spec.model, "trials": have,
                            "file": out.name})
            continue
        if have:
            console.print(f"  [green]resume[/green] {spec.name}: {have}/{n_trials} done, continuing")
        try:
            res = run_model(spec, exp, exp.prompts, exp.pairs, out_path=out, overwrite=overwrite)
            written.append({"model": spec.name, "hf": spec.model, "trials": len(res),
                            "file": out.name})
            console.print(f"  wrote {len(res)} trials -> {out}")
        except Exception as exc:  # noqa: BLE001 - skip a failed model, keep the batch going
            written.append({"model": spec.name, "hf": spec.model, "trials": 0,
                            "error": f"{type(exc).__name__}: {exc}"})
            console.print(f"  [red]FAILED[/red] {spec.name}: {type(exc).__name__}: {exc}")

    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiment": json.loads(exp.model_dump_json()),
        "models": written,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.print(f"manifest -> {run_dir / 'manifest.json'}")


@app.command()
def judge(
    in_: Path = typer.Option(..., "--in", help="a run dir or a single .raw.jsonl"),
    experiment: Path = typer.Option("configs/experiment.yaml"),
):
    """Phase 2: judge every model in the run with one Gemma load; archive verdicts."""
    from . import judge as judging
    from .runner import append_index, raw_files, read_results, write_results

    exp = load_experiment(experiment)
    files = raw_files(in_)
    if not files:
        console.print(f"[yellow]no .raw.jsonl found under {in_}[/yellow]")
        raise typer.Exit(1)

    all_res = []
    for f in files:
        all_res.extend(read_results(f))
    console.rule(f"[magenta]judge[/magenta] ({exp.judge.model}) — {len(all_res)} trials")

    judged = judging.judge_results(all_res, exp.judge)  # single Gemma load

    out_dir = Path(in_) if Path(in_).is_dir() else Path(in_).parent
    by_model = defaultdict(list)
    for r in judged:
        by_model[r.model].append(r)
    for model, rs in by_model.items():
        write_results(rs, out_dir / f"{model}.judged.jsonl")

    run_id = out_dir.name
    append_index(INDEX_CSV, run_id, judged)
    console.print(f"judged {len(judged)} trials -> {out_dir} (index: {INDEX_CSV})")


@app.command()
def rescore(
    in_: Path = typer.Option(..., "--in", help="a run dir or a single .jsonl"),
):
    """Re-apply the text-evidence + thinking detectors over existing results (no GPU)."""
    from . import evidence, thinking
    from .runner import read_results, write_results

    p = Path(in_)
    files = sorted(p.glob("**/*.jsonl")) if p.is_dir() else [p]
    total = flagged = aware = thought = 0
    for f in files:
        results = read_results(f)
        for r in results:
            r.text_evidence = evidence.detect(r.reaction_text)
            # preserve template-level default_on (only known at generation time)
            r.thinking = thinking.detect(r.reaction_text, default_on=r.thinking.default_on)
            flagged += int(r.text_evidence.flagged)
            aware += int(r.text_evidence.aware)
            thought += int(r.thinking.detected)
            total += 1
        write_results(results, f)
    console.print(
        f"re-scored {total} trials across {len(files)} files — "
        f"{flagged} text-flagged ({aware} substitution-aware), {thought} with thinking in output"
    )


@app.command()
def report(in_: Path = typer.Option(..., "--in", help="a run dir, results/runs (all), or a file")):
    """Aggregate results into a per-model table (accepts a file or a directory)."""
    from .report import print_report

    print_report(in_)


if __name__ == "__main__":
    app()
