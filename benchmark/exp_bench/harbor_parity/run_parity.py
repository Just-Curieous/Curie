"""Run the EXP-Bench harness on a list of task ids for parity with the Harbor adapter.

Drives evaluation/main_eval.py (one container per paper and iteration) with the task
records, pinned repository commits and ground-truth sources of the adapter's task bundle.
Task ids are <conference>-<paper_id>-<task_index>, where task_index is the question's
position in upstream's <paper_id>_complete_final.json. See harbor_parity/README.md.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EXP_BENCH_DIR = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PARITY_DIR = Path("outputs/harbor_parity")  # relative paths also resolve inside the container (cwd /)
TASK_ID_RE = re.compile(r"^(iclr2024|neurips2024)-(\d+)-(\d+)$")
SOURCE_CACHE_DIR = Path.home() / ".cache" / "harbor-exp-bench-adapter"  # shared with the adapter
MOUNTED_DIRS = {"evaluation", "outputs", "logs", "helper", "workspace", "inspect_agent"}


def log(msg: str) -> None:
    print(f"[harbor-parity] {msg}", flush=True)


def read_task_ids(values: list[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        path = Path(value)
        for line in path.read_text().splitlines() if path.is_file() else [value]:
            line = line.split("#", 1)[0].strip()
            if line and line not in ids:
                if not TASK_ID_RE.match(line):
                    sys.exit(f"Invalid task id {line!r}")
                ids.append(line)
    return ids


def model_identifier(env_file: Path) -> str:
    """evaluation/parallel_eval.py:extract_model_identifier (names the output folder)."""
    for line in env_file.read_text().splitlines():
        if match := re.match(r'export\s+MODEL="([^"]+)"', line):
            return match.group(1).replace("/", "-").replace(":", "-").replace(".", "-")
    sys.exit(f'{env_file} must contain export MODEL="..."')


def fetch_source(repo: str, commit: str, path: str, sha256: str, cache_dir: Path) -> str:
    """A ground-truth source file at the pinned commit, checked against the bundle's sha256."""
    rel = path.removeprefix("/workspace").lstrip("/")
    cached = cache_dir / repo / commit / rel
    if not cached.exists():
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
                break
            except urllib.error.URLError:
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    text = cached.read_bytes().decode("utf-8", errors="replace")
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        sys.exit(f"sha256 mismatch for {repo}@{commit}:{rel}")
    return text


def write_paper_files(conference: str, paper_id: str, records: list[dict], selected: list[int],
                      cache_dir: Path) -> tuple[str, str]:
    """The paper's full question list and the pinned ground-truth sources of the selected tasks."""
    records = sorted(records, key=lambda r: r["task_index"])
    out_dir = PARITY_DIR / "tasks" / conference / paper_id
    (EXP_BENCH_DIR / out_dir).mkdir(parents=True, exist_ok=True)
    tasks_file = out_dir / f"{paper_id}_complete_final.json"
    (EXP_BENCH_DIR / tasks_file).write_text(json.dumps({"questions": [r["task_data"] for r in records]}, indent=1))
    scripts = {
        str(r["task_index"]): {
            path: fetch_source(r["repo"], r["commit"], path, sha, cache_dir) if sha else None
            for path, sha in r["setup_script_sha256"].items()
        }
        for r in records if r["task_index"] in selected
    }
    scripts_file = out_dir / f"{paper_id}_setup_scripts.json"
    (EXP_BENCH_DIR / scripts_file).write_text(json.dumps(scripts, indent=1))
    return str(tasks_file), str(scripts_file)


def write_setup_apis_dir() -> str:
    """setup_apis_exp/env.sh as the adapter writes it (credentials for experiments, may be empty)."""
    out_dir = PARITY_DIR / "setup_apis_exp"
    (EXP_BENCH_DIR / out_dir).mkdir(parents=True, exist_ok=True)
    key = os.environ.get("EXP_BENCH_TASK_OPENAI_API_KEY", "")
    base = os.environ.get("EXP_BENCH_TASK_OPENAI_BASE_URL", "")
    (EXP_BENCH_DIR / out_dir / "env.sh").write_text(
        f'export OPENAI_API_KEY="{key}"\nexport OPENAI_BASE_URL="{base}"\nexport OPENAI_API_BASE="{base}"\n'
    )
    return str(out_dir)


def output_file(args, conference, paper_id, model_id, idx, iteration, mode) -> Path:
    suffix = "eval_gen" if mode == "generate" else "eval_judge"
    folder = Path(args.output_root) / conference / paper_id / args.agent_name / model_id
    return EXP_BENCH_DIR / folder / f"{paper_id}_task_index_{idx}_iter_{iteration}_duration_{args.max_duration}_{suffix}.json"


def is_done(path: Path, mode: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return mode == "generate" or "execution_success" in json.loads(path.read_text())


def build_config(args, conference, paper_id, rec0, indices, iterations, model_id, tasks_file, scripts_file, setup_apis_dir) -> dict:
    """The upstream config fields the harness reads, plus the Harbor parity keys."""
    out = Path(args.output_root) / conference / paper_id / args.agent_name / model_id
    logs = Path(args.output_root) / "logs" / conference / paper_id / args.agent_name / model_id
    for folder in (out, logs):
        (EXP_BENCH_DIR / folder).mkdir(parents=True, exist_ok=True)
    prompts = "evaluation/prompts"
    return {
        "parallelization_factor": args.parallel,
        "iterations": max(iterations),
        "llm_config_filename": args.agent_env,
        "agent_name": args.agent_name,
        "max_duration_per_task_in_hours": args.max_duration,
        "mode": args.mode,
        "docker_image": args.docker_image,
        "dockerfile_name": "ExpDockerfile_default",
        "eval_gen_prompt": f"{prompts}/eval_gen_prompt.txt",
        "eval_judge_prompt": f"{prompts}/eval_judge_prompt.txt",
        "eval_judge_setup_prompt": f"{prompts}/eval_judge_setup_prompt.txt",
        "eval_judge_setup_partial_prompt": f"{prompts}/eval_judge_setup_partial_prompt.txt",
        "eval_judge_setup_monitor_prompt": f"{prompts}/eval_judge_setup_monitor_prompt.txt",
        "do_exec_check": not args.no_exec_check,
        "llm_judge_config_filename": args.judge_env or "",
        "judge_agent_name": "",
        "paper_id": int(paper_id),
        "input_paper_tasks_filename": tasks_file,
        "output_folder": str(out),
        "output_log_folder": str(logs),
        "github_url": rec0["github_url"],
        "specific_tasks": [[paper_id, idx, args.max_duration] for idx in indices],
        # Harbor parity
        "iteration_list": iterations,
        "pinned_commit": rec0["commit"],
        "pinned_setup_scripts_filename": scripts_file,
        "setup_apis_source": setup_apis_dir,
        "prune_docker_after_paper": not args.no_prune,
        "claude_code_version": args.claude_code_version,
        "claude_code_flags": args.claude_code_flags,
        "sandbox_docker_args": ["--cpus", str(args.sandbox_cpus), "--memory", args.sandbox_memory],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["generate", "judge"])
    parser.add_argument("--task-ids", nargs="+", default=[str(HERE / "data" / "parity_task_ids.txt")],
                        help="Files with one task id per line and/or literal task ids.")
    parser.add_argument("--bundle", type=Path, default=HERE / "data" / "exp_bench_tasks.json.gz")
    parser.add_argument("--agent-env", required=True, help='Agent env file (export MODEL="..." plus credentials).')
    parser.add_argument("--judge-env", help="Judge env file (MODEL, OPENAI_API_KEY, OPENAI_BASE_URL); judge mode only.")
    parser.add_argument("--iteration", type=int, nargs="+", default=[1], help="Iterations (runs) to do, e.g. 1 2 3.")
    parser.add_argument("--max-duration", type=float, default=0.5, help="Agent time limit in hours (upstream default).")
    parser.add_argument("--parallel", type=int, default=6, help="Tasks of one paper run in parallel.")
    parser.add_argument("--agent-name", default="claude-code")
    parser.add_argument("--claude-code-version", default="2.1.330", help="Same as --ak version= on the Harbor side.")
    parser.add_argument("--claude-code-flags", default="--permission-mode=bypassPermissions",
                        help="CLI flags of Harbor's claude-code agent for the same kwargs.")
    parser.add_argument("--sandbox-cpus", type=int, default=4, help="As cpus in the adapter's task.toml.")
    parser.add_argument("--sandbox-memory", default="16g", help="As memory_mb in the adapter's task.toml.")
    parser.add_argument("--docker-image", default="exp-bench-image")
    parser.add_argument("--output-root", default="outputs/evaluation")
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE_DIR)
    parser.add_argument("--no-exec-check", action="store_true")
    parser.add_argument("--no-prune", action="store_true", help="Skip upstream's docker prune after each paper.")
    parser.add_argument("--retry", action="store_true",
                        help="Re-run (paper, iteration) pairs already attempted (default: one attempt per task).")
    args = parser.parse_args()

    os.chdir(EXP_BENCH_DIR)
    if args.mode == "judge" and not args.judge_env:
        sys.exit("judge mode needs --judge-env")
    for env_file in filter(None, [args.agent_env, args.judge_env]):
        # Sourced inside the container, where only these directories are mounted.
        if not Path(env_file).is_file() or Path(env_file).is_absolute() or Path(env_file).parts[0] not in MOUNTED_DIRS:
            sys.exit(f"{env_file} must be an existing relative path under one of {sorted(MOUNTED_DIRS)}")
    if subprocess.run(["docker", "image", "inspect", args.docker_image], capture_output=True).returncode != 0:
        sys.exit(f"Docker image {args.docker_image} not found; build it with harbor_parity/build_image.sh")

    raw = gzip.decompress(args.bundle.read_bytes())
    bundle = json.loads(raw)
    by_id = {t["task_id"]: t for t in bundle["tasks"]}
    task_ids = read_task_ids(args.task_ids)
    if unknown := [t for t in task_ids if t not in by_id]:
        sys.exit(f"Task ids not in the bundle: {unknown}")
    model_id = model_identifier(Path(args.agent_env))
    log(f"{len(task_ids)} tasks, iterations {args.iteration}, model {model_id}, bundle sha256 {hashlib.sha256(raw).hexdigest()}")

    papers: dict[tuple[str, str], list[int]] = defaultdict(list)
    all_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task_id in task_ids:
        papers[(by_id[task_id]["conference"], by_id[task_id]["paper_id"])].append(by_id[task_id]["task_index"])
    for rec in bundle["tasks"]:
        all_records[(rec["conference"], rec["paper_id"])].append(rec)

    setup_apis_dir = write_setup_apis_dir()
    config_dir = Path("outputs/evaluation/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for (conference, paper_id), indices in papers.items():
        indices = sorted(indices)
        todo = [it for it in args.iteration
                if not all(is_done(output_file(args, conference, paper_id, model_id, i, it, args.mode), args.mode) for i in indices)]
        if args.mode == "judge":  # judge.py skips tasks without a generation output
            todo = [it for it in todo
                    if any(output_file(args, conference, paper_id, model_id, i, it, "generate").exists() for i in indices)]
        markers = {it: EXP_BENCH_DIR / PARITY_DIR / "attempts" / args.mode / f"{args.agent_name}_{model_id}"
                   / f"{conference}_{paper_id}_iter_{it}_duration_{args.max_duration}" for it in todo}
        if not args.retry:
            todo = [it for it in todo if not markers[it].exists()]
        if not todo:
            log(f"{conference}/{paper_id}: nothing to do")
            continue
        tasks_file, scripts_file = write_paper_files(conference, paper_id, all_records[(conference, paper_id)],
                                                     indices, args.source_cache)
        rec0 = by_id[f"{conference}-{paper_id}-{indices[0]}"]
        config = build_config(args, conference, paper_id, rec0, indices, todo, model_id, tasks_file, scripts_file, setup_apis_dir)
        config_path = config_dir / f"harbor_parity_{conference}_{paper_id}_{args.mode}_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        log(f"{conference}/{paper_id} tasks {indices} iterations {todo} @ {rec0['commit'][:12]}")
        for it in todo:
            markers[it].parent.mkdir(parents=True, exist_ok=True)
            markers[it].write_text(datetime.now().isoformat(timespec="seconds"))
        env = dict(os.environ, PYTHONPATH=str(EXP_BENCH_DIR))
        if subprocess.run([sys.executable, "evaluation/main_eval.py", f"--task_config={config_path}"], env=env).returncode:
            failures.append(f"{conference}/{paper_id}")
    log("done" + (f"; failed papers: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
