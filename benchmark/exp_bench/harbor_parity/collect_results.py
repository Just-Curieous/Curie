"""Turn upstream EXP-Bench judge outputs into Harbor reward values and compare with Harbor.

Per task and run, the upstream ``*_eval_judge.json`` is converted with exactly the
definitions of the Harbor adapter's ``tests/judge.py`` (``/logs/verifier/reward.json``):

  design          = clamp(float(design_score), 0, 100) / 100   (non-numeric -> 0)
  implementation  = clamp(float(setup_score), 0, 100) / 100     (non-numeric -> 0)
  conclusion      = 1 if conclusion_score.strip().lower() == "correct" else 0
  execution       = 1 if execution_success is True else 0
  implementation_and_execution = implementation == 1 and execution == 1
  all_correct     = design == 1 and implementation == 1 and conclusion == 1
  all_correct_and_execution = all_correct and execution == 1
  monitor_pass    = 0 if the log monitor flagged the run (upstream then writes
                    setup_monitor_comprehensive_reason and zero scores), else 1
  reward          = conclusion

A missing judge file (or a Harbor trial without rewards) counts as 0 for every metric
by default and is listed; use ``--missing exclude`` to drop such tasks from that run.

Statistics, per metric (values in [0, 1], printed as percentages):
  run mean        mean over tasks within one run
  mean +/- SEM    mean of the run means, sample standard error (ddof=1) across runs
  range overlap   whether [min, max] of the original run means and of the Harbor run
                  means overlap
  per-task MSE    mean over tasks of (original per-task mean - Harbor per-task mean)^2,
                  where per-task means average that task's runs on each side

Usage (from benchmark/exp_bench):
  python harbor_parity/collect_results.py --agent-env evaluation/setup/env-parity-agent.sh \
      --iterations 1 2 3 --harbor-jobs /path/to/jobs/run1 /path/to/jobs/run2 /path/to/jobs/run3 \
      --out outputs/harbor_parity/parity_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXP_BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TASK_IDS = Path(__file__).resolve().parent / "data" / "parity_task_ids.txt"
METRICS = [
    "design",
    "implementation",
    "conclusion",
    "execution",
    "implementation_and_execution",
    "all_correct",
    "all_correct_and_execution",
    "monitor_pass",
]
TASK_ID_RE = re.compile(r"^(iclr2024|neurips2024)-(\d+)-(\d+)$")


# --------------------------------------------------------------------------- inputs


def read_task_ids(values: list[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        path = Path(value)
        for line in (path.read_text().splitlines() if path.is_file() else [value]):
            line = line.split("#", 1)[0].strip()
            if line and line not in ids:
                ids.append(line)
    return ids


def model_identifier(env_file: Path) -> str:
    for line in env_file.read_text().splitlines():
        match = re.match(r'export\s+MODEL="([^"]+)"', line)
        if match:
            return match.group(1).replace("/", "-").replace(":", "-").replace(".", "-")
    sys.exit(f'{env_file} has no export MODEL="..." line')


def to_fraction(score) -> float:
    """Harbor adapter tests/judge.py:to_fraction."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return max(0.0, min(100.0, value)) / 100.0


def rewards_from_judge(judged: dict) -> dict[str, float]:
    """Harbor adapter tests/judge.py reward definitions applied to an upstream eval_judge.json."""
    monitor_flagged = "setup_monitor_comprehensive_reason" in judged
    design = to_fraction(judged.get("design_score"))
    implementation = to_fraction(judged.get("setup_score"))
    conclusion = 1.0 if str(judged.get("conclusion_score", "")).strip().lower() == "correct" else 0.0
    execution = 1.0 if judged.get("execution_success") is True else 0.0
    all_correct = 1.0 if (design == 1.0 and implementation == 1.0 and conclusion == 1.0) else 0.0
    return {
        "reward": conclusion,
        "design": design,
        "implementation": implementation,
        "conclusion": conclusion,
        "execution": execution,
        "implementation_and_execution": 1.0 if (implementation == 1.0 and execution == 1.0) else 0.0,
        "all_correct": all_correct,
        "all_correct_and_execution": 1.0 if (all_correct == 1.0 and execution == 1.0) else 0.0,
        "monitor_pass": 0.0 if monitor_flagged else 1.0,
    }


def load_original(args, task_ids: list[str]) -> list[dict]:
    """One dict per iteration: {"label", "values": {task_id: rewards | None}, "notes": {...}}."""
    model_id = args.model_id or model_identifier(Path(args.agent_env))
    root = Path(args.original_root)
    if not root.is_absolute():
        root = EXP_BENCH_DIR / root
    runs = []
    for iteration in args.iterations:
        values, notes = {}, {}
        for task_id in task_ids:
            conference, paper_id, idx = TASK_ID_RE.match(task_id).groups()
            folder = root / conference / paper_id / args.agent_name / model_id
            judge = folder / f"{paper_id}_task_index_{idx}_iter_{iteration}_duration_{args.duration}_eval_judge.json"
            gen = folder / f"{paper_id}_task_index_{idx}_iter_{iteration}_duration_{args.duration}_eval_gen.json"
            if judge.exists():
                judged = json.loads(judge.read_text())
                values[task_id] = rewards_from_judge(judged)
                if "setup_monitor_comprehensive_reason" in judged:
                    notes[task_id] = "log monitor flagged forbidden operations (all scores 0)"
                elif "execution_success" not in judged:
                    notes[task_id] = "judged without execution check"
                if gen.exists() and "no_answer" in json.loads(gen.read_text()):
                    notes[task_id] = "agent gave no answer (no_answer)"
            else:
                values[task_id] = None
                notes[task_id] = "missing eval_judge.json" + ("" if gen.exists() else " and eval_gen.json")
        runs.append({"label": f"original iter {iteration}", "values": values, "notes": notes})
    return runs


def harbor_task_id(result: dict, trial_dir: Path) -> str | None:
    for name in (result.get("task_name") or "", trial_dir.name):
        match = re.search(r"((?:iclr2024|neurips2024)-\d+-\d+)", name)
        if match:
            return match.group(1)
    return None


def load_harbor(job_dirs: list[Path], task_ids: list[str], agent_timeout: str = "keep") -> list[dict]:
    """Each job directory is one run; a job with k trials per task gives k runs (ordered by
    trial start time)."""
    runs = []
    for job_dir in job_dirs:
        trials: dict[str, list[tuple[str, dict | None, str]]] = defaultdict(list)
        for result_path in sorted(job_dir.glob("*/result.json")):
            result = json.loads(result_path.read_text())
            if "trial_name" not in result:
                continue  # job-level result.json lives at the job root, not in a trial dir
            task_id = harbor_task_id(result, result_path.parent)
            if task_id is None:
                continue
            rewards = (result.get("verifier_result") or {}).get("rewards")
            note = ""
            exc_type = (result.get("exception_info") or {}).get("exception_type")
            if exc_type == "AgentTimeoutError" and agent_timeout == "zero":
                # Upstream never judges a timed-out task (call_oh_with_prompt raises), so this
                # scores Harbor timeouts the way the original side ends up scoring them.
                rewards, note = {m: 0.0 for m in METRICS}, "agent timeout scored as 0 (--harbor-agent-timeout zero)"
            elif exc_type == "AgentTimeoutError":
                note = "agent timeout (graded by Harbor)"
            if not rewards:
                reward_json = result_path.parent / "verifier" / "reward.json"
                if reward_json.exists():
                    rewards = json.loads(reward_json.read_text())
                else:
                    note = f"no rewards ({exc_type or 'verifier did not run'})"
            trials[task_id].append((result.get("started_at") or "", rewards or None, note))
        n_attempts = max((len(v) for v in trials.values()), default=0)
        for attempt in range(max(n_attempts, 1)):
            values, notes = {}, {}
            for task_id in task_ids:
                attempts = sorted(trials.get(task_id, []), key=lambda t: t[0])
                if attempt < len(attempts):
                    _, rewards, note = attempts[attempt]
                    values[task_id] = {m: float(rewards.get(m, 0.0)) for m in METRICS} if rewards else None
                    if note:
                        notes[task_id] = note
                else:
                    values[task_id] = None
                    notes[task_id] = "no trial"
            label = f"harbor {job_dir.name}" + (f" attempt {attempt + 1}" if n_attempts > 1 else "")
            runs.append({"label": label, "values": values, "notes": notes})
    return runs


# --------------------------------------------------------------------------- stats


def run_mean(run: dict, metric: str, missing: str) -> float | None:
    vals = []
    for rewards in run["values"].values():
        if rewards is None:
            if missing == "zero":
                vals.append(0.0)
        else:
            vals.append(rewards[metric])
    return sum(vals) / len(vals) if vals else None


def mean_sem(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return mean, sem


def per_task_means(runs: list[dict], metric: str, missing: str) -> dict[str, float]:
    out = {}
    task_ids = runs[0]["values"].keys() if runs else []
    for task_id in task_ids:
        vals = []
        for run in runs:
            rewards = run["values"].get(task_id)
            if rewards is None:
                if missing == "zero":
                    vals.append(0.0)
            else:
                vals.append(rewards[metric])
        if vals:
            out[task_id] = sum(vals) / len(vals)
    return out


def fmt(mean: float | None, sem: float | None) -> str:
    if mean is None:
        return "n/a"
    return f"{100 * mean:.1f}%" + (f" +/- {100 * sem:.1f}%" if sem is not None else "")


def summarize(original: list[dict], harbor: list[dict], missing: str) -> dict:
    summary = {}
    for metric in METRICS:
        entry = {}
        for side, runs in (("original", original), ("harbor", harbor)):
            means = [m for m in (run_mean(r, metric, missing) for r in runs) if m is not None]
            mean, sem = mean_sem(means)
            entry[side] = {"runs": means, "mean": mean, "sem": sem, "formatted": fmt(mean, sem)}
        if original and harbor:
            o, h = entry["original"]["runs"], entry["harbor"]["runs"]
            entry["ranges_overlap"] = bool(o and h and max(min(o), min(h)) <= min(max(o), max(h)))
            ot, ht = per_task_means(original, metric, missing), per_task_means(harbor, metric, missing)
            common = sorted(set(ot) & set(ht))
            diffs = {t: ot[t] - ht[t] for t in common}
            entry["per_task_mse"] = sum(d * d for d in diffs.values()) / len(common) if common else None
            entry["per_task_rmse_pp"] = 100 * math.sqrt(entry["per_task_mse"]) if common else None
            entry["n_tasks_compared"] = len(common)
            entry["per_task_diff"] = diffs
        summary[metric] = entry
    return summary


def per_task_table(runs: list[dict]) -> dict[str, dict]:
    table = {}
    for run in runs:
        for task_id, rewards in run["values"].items():
            table.setdefault(task_id, {})[run["label"]] = rewards
    return table


def print_report(summary: dict, original: list[dict], harbor: list[dict]) -> None:
    print(f"original runs: {[r['label'] for r in original]}")
    print(f"harbor runs:   {[r['label'] for r in harbor]}")
    header = f"{'metric':30s} {'original':>20s} {'harbor':>20s} {'overlap':>8s} {'MSE':>8s} {'RMSE(pp)':>9s}"
    print(header)
    print("-" * len(header))
    for metric, e in summary.items():
        overlap = "" if "ranges_overlap" not in e else ("yes" if e["ranges_overlap"] else "NO")
        mse = "" if e.get("per_task_mse") is None else f"{e['per_task_mse']:.4f}"
        rmse = "" if e.get("per_task_rmse_pp") is None else f"{e['per_task_rmse_pp']:.1f}"
        print(f"{metric:30s} {e['original']['formatted']:>20s} {e['harbor']['formatted']:>20s} {overlap:>8s} {mse:>8s} {rmse:>9s}")
    for side in (original, harbor):
        for run in side:
            for task_id, note in run["notes"].items():
                print(f"  note [{run['label']}] {task_id}: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-ids", nargs="+", default=[str(DEFAULT_TASK_IDS)])
    parser.add_argument("--original-root", default="outputs/evaluation")
    parser.add_argument("--agent-name", default="openhands")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--agent-env", help="Agent env file used for generation (gives the output folder name).")
    model.add_argument("--model-id", help="Output folder model name, e.g. openai-gpt-4o-mini.")
    parser.add_argument("--duration", default="0.5", help="max_duration_per_task_in_hours as written in file names.")
    parser.add_argument("--iterations", type=int, nargs="*", default=[1], help="Original-side iterations (runs).")
    parser.add_argument("--harbor-jobs", type=Path, nargs="*", default=[], help="Harbor job directories (one run each).")
    parser.add_argument("--missing", choices=["zero", "exclude"], default="zero")
    parser.add_argument("--harbor-agent-timeout", choices=["keep", "zero"], default="keep",
                        help="Harbor grades runs that hit the agent timeout; upstream writes no output for them "
                             "(counted as missing). 'zero' scores Harbor timeouts as 0 to match.")
    parser.add_argument("--out", type=Path, help="Write the full results (summary + per-task values) as JSON.")
    args = parser.parse_args()
    if args.iterations and not (args.agent_env or args.model_id):
        parser.error("--agent-env or --model-id is required for the original side")
    args.duration = str(float(args.duration)) if args.duration.replace(".", "", 1).isdigit() else args.duration

    task_ids = read_task_ids(args.task_ids)
    original = load_original(args, task_ids) if args.iterations else []
    harbor = load_harbor(args.harbor_jobs, task_ids, args.harbor_agent_timeout)
    summary = summarize(original, harbor, args.missing)
    print_report(summary, original, harbor)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "task_ids": task_ids,
            "missing_policy": args.missing,
            "summary": summary,
            "original_per_task": per_task_table(original),
            "harbor_per_task": per_task_table(harbor),
            "notes": {r["label"]: r["notes"] for r in original + harbor},
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
