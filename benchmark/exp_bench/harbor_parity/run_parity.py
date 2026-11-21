"""Run the upstream EXP-Bench harness on an explicit list of task ids (Harbor parity).

This drives the unmodified upstream pipeline (evaluation/main_eval.py -> one Docker
container per paper and iteration -> evaluation/eval.py or evaluation/judge.py), with
the task records, pinned repository commits and ground-truth source files taken from
the same task bundle the Harbor adapter is generated from:

* task ids are ``<conference>-<paper_id>-<task_index>``; ``task_index`` is the position
  of the question in upstream's ``<paper_id>_complete_final.json``;
* each paper's questions are written to
  ``outputs/harbor_parity/tasks/<conference>/<paper_id>/<paper_id>_complete_final.json``
  from the bundle (identical to upstream's ``outputs/logs`` files; checked when present);
* each run clones the paper repository at the bundle's pinned commit with the Harbor
  adapter's shallow fetch (``pinned_commit`` in the per-paper config);
* the judge reads ``setup_ground_truth_scripts`` from the bundle
  (``pinned_setup_scripts_filename``) instead of the repositories' main/master branch.

Papers are processed one after another, as upstream ``evaluation/parallel_eval.py`` does;
tasks of one paper run in parallel inside its container (``--parallel``).

Examples (from benchmark/exp_bench):

    python harbor_parity/run_parity.py generate --agent-env evaluation/setup/env-parity-agent.sh --iterations 1
    python harbor_parity/run_parity.py judge --agent-env evaluation/setup/env-parity-agent.sh \
        --judge-env evaluation/setup/env-parity-judge.sh --iterations 1
    python harbor_parity/run_parity.py generate --dry-run --task-ids neurips2024-98316-5 \
        --agent-env evaluation/setup/env-parity-agent.sh
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
DEFAULT_BUNDLE = HERE / "data" / "exp_bench_tasks.json.gz"
DEFAULT_TASK_IDS = HERE / "data" / "parity_task_ids.txt"
# Relative to benchmark/exp_bench; outputs/ is mounted at /outputs in the upstream
# container, whose working directory is /, so the same relative paths work inside it.
PARITY_DIR = Path("outputs/harbor_parity")
TASK_ID_RE = re.compile(r"^(iclr2024|neurips2024)-(\d+)-(\d+)$")
# Same cache as the Harbor adapter (src/exp_bench/adapter.py:DEFAULT_CACHE_DIR).
SOURCE_CACHE_DIR = Path.home() / ".cache" / "harbor-exp-bench-adapter"
MOUNTED_DIRS = {"evaluation", "outputs", "logs", "helper", "workspace", "inspect_agent"}


def log(msg: str) -> None:
    print(f"[harbor-parity] {msg}", flush=True)


# --------------------------------------------------------------------------- inputs


def load_bundle(path: Path) -> tuple[dict, str]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_task_ids(values: list[str]) -> list[str]:
    """Task ids from files and/or literal ids, in order, without duplicates."""
    ids: list[str] = []
    for value in values:
        path = Path(value)
        lines = path.read_text().splitlines() if path.is_file() else [value]
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if line and line not in ids:
                if not TASK_ID_RE.match(line):
                    sys.exit(f"Invalid task id {line!r} (expected <conference>-<paper_id>-<task_index>)")
                ids.append(line)
    return ids


def read_env_file(path: Path) -> dict[str, str]:
    """Same parsing as helper/utils.py:parse_env_string."""
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export"):
            line = line.replace("export", "", 1).strip()
        if "=" in line:
            key, value = line.split("#")[0].strip().split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
    return env


def model_identifier(env_file: Path) -> str:
    """evaluation/parallel_eval.py:extract_model_identifier (names the output folder)."""
    for line in env_file.read_text().splitlines():
        match = re.match(r'export\s+MODEL="([^"]+)"', line)
        if match:
            return match.group(1).replace("/", "-").replace(":", "-").replace(".", "-")
    sys.exit(f'{env_file} must contain a line export MODEL="..." (upstream requirement)')


def paper_details(conference: str) -> dict[str, dict]:
    """pdf_url per paper from upstream's paper-details JSONL (informational only)."""
    details = {}
    for path in (EXP_BENCH_DIR / "logs" / conference).glob("*.json"):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                details[str(record["id"])] = record
    return details


# --------------------------------------------------------------------------- files


def github_raw_path(path: str) -> str:
    """Path handling of upstream helper/utils.py:get_file_from_github."""
    if path.startswith("/workspace"):
        path = path[len("/workspace"):]
    return path.lstrip("/")


def fetch_source(repo: str, commit: str, path: str, expected_sha256: str, cache_dir: Path) -> str:
    """Ground-truth source at the pinned commit, verified against the bundle's sha256.
    Same download, decoding, hash and cache layout as the Harbor adapter's fetch_source."""
    rel = github_raw_path(path)
    cached = cache_dir / repo / commit / rel
    if cached.exists():
        data = cached.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"
        data = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    raise FileNotFoundError(url) from e
                time.sleep(3 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"Could not download {url}")
    text = data.decode("utf-8", errors="replace")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256:
        raise ValueError(f"sha256 mismatch for {repo}@{commit}:{rel}")
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    return text


def setup_scripts_for(record: dict, cache_dir: Path) -> dict[str, str | None]:
    """{source path: content} in task order. Full bundles carry the contents
    (setup_scripts); the adapter's slim bundle carries setup_script_sha256 only."""
    if "setup_scripts" in record:
        return record["setup_scripts"]
    return {
        path: (fetch_source(record["repo"], record["commit"], path, sha, cache_dir) if sha else None)
        for path, sha in record["setup_script_sha256"].items()
    }


def write_paper_files(conference: str, paper_id: str, records: list[dict], selected: list[int],
                      cache_dir: Path) -> tuple[str, str]:
    """Per-paper task list (all questions, so task_index keeps its meaning) and the pinned
    ground-truth sources of the selected tasks, both from the bundle."""
    records = sorted(records, key=lambda r: r["task_index"])
    if [r["task_index"] for r in records] != list(range(len(records))):
        sys.exit(f"Bundle is missing task indices for {conference}/{paper_id}")
    out_dir = PARITY_DIR / "tasks" / conference / paper_id
    (EXP_BENCH_DIR / out_dir).mkdir(parents=True, exist_ok=True)

    questions = [r["task_data"] for r in records]
    upstream_file = EXP_BENCH_DIR / "outputs" / "logs" / conference / paper_id / f"{paper_id}_complete_final.json"
    if upstream_file.exists() and json.loads(upstream_file.read_text())["questions"] != questions:
        log(f"WARNING: bundle questions differ from {upstream_file.relative_to(EXP_BENCH_DIR)}; using the bundle")
    tasks_file = out_dir / f"{paper_id}_complete_final.json"
    (EXP_BENCH_DIR / tasks_file).write_text(json.dumps({"questions": questions}, indent=1, ensure_ascii=False))

    scripts_file = out_dir / f"{paper_id}_setup_scripts.json"
    scripts = {str(r["task_index"]): setup_scripts_for(r, cache_dir) for r in records if r["task_index"] in selected}
    (EXP_BENCH_DIR / scripts_file).write_text(json.dumps(scripts, indent=1, ensure_ascii=False))
    return str(tasks_file), str(scripts_file)


def write_setup_apis_dir() -> str:
    """Contents of /workspace/setup_apis_exp/ for the agent.

    Upstream copies its own credential files (evaluation/setup/) into the repository.
    The Harbor adapter instead writes setup_apis_exp/env.sh exporting OPENAI_API_KEY /
    OPENAI_BASE_URL from EXP_BENCH_TASK_OPENAI_API_KEY / EXP_BENCH_TASK_OPENAI_BASE_URL
    (empty unless set). Same here, with the values resolved from this process's environment.
    """
    key = os.environ.get("EXP_BENCH_TASK_OPENAI_API_KEY", "")
    base = os.environ.get("EXP_BENCH_TASK_OPENAI_BASE_URL", "")
    out_dir = PARITY_DIR / "setup_apis_exp"
    (EXP_BENCH_DIR / out_dir).mkdir(parents=True, exist_ok=True)
    (EXP_BENCH_DIR / out_dir / "env.sh").write_text(
        "# LLM credentials for experiments that need to call an LLM API (may be empty).\n"
        f'export OPENAI_API_KEY="{key}"\n'
        f'export OPENAI_BASE_URL="{base}"\n'
        f'export OPENAI_API_BASE="{base}"\n'
    )
    return str(out_dir)


# --------------------------------------------------------------------------- configs


def output_folders(args, conference: str, paper_id: str, model_id: str) -> tuple[str, str]:
    """Same layout as evaluation/parallel_eval.py:convert_to_paper_dir_path."""
    root = Path(args.output_root)
    out = root / conference / paper_id / args.agent_name / model_id
    logs = root / "logs" / conference / paper_id / args.agent_name / model_id
    return str(out), str(logs)


def eval_file(output_folder: str, paper_id: str, idx: int, iteration: int, duration: float, mode: str) -> Path:
    """evaluation/utils.py:get_relative_output_path_eval."""
    suffix = "eval_gen" if mode == "generate" else "eval_judge"
    return EXP_BENCH_DIR / output_folder / f"{paper_id}_task_index_{idx}_iter_{iteration}_duration_{duration}_{suffix}.json"


def is_done(output_folder: str, paper_id: str, idx: int, iteration: int, duration: float, mode: str) -> bool:
    path = eval_file(output_folder, paper_id, idx, iteration, duration, mode)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if mode == "judge":
        return "execution_success" in json.loads(path.read_text())
    return True


def build_config(args, mode, conference, paper_id, rec0, indices, iterations, model_id, files, manifest) -> dict:
    tasks_file, scripts_file, setup_apis_dir = files
    output_folder, output_log_folder = output_folders(args, conference, paper_id, model_id)
    # As evaluation/parallel_eval.py:create_main_task_config does (eval.py expects them).
    (EXP_BENCH_DIR / output_folder).mkdir(parents=True, exist_ok=True)
    (EXP_BENCH_DIR / output_log_folder).mkdir(parents=True, exist_ok=True)
    details = paper_details(conference).get(paper_id, {})
    duration = args.max_duration
    config = {
        # Fields of evaluation/configs/parallel_eval_*_config_template.json
        "input_conference_tasks_folder": f"outputs/logs/{conference}",
        "input_conference_paper_details_filename": next(
            (str(p.relative_to(EXP_BENCH_DIR)) for p in (EXP_BENCH_DIR / "logs" / conference).glob("*.json")), ""
        ),
        "parallelization_factor": args.parallel,
        "iterations": max(iterations),
        "llm_config_filename": args.agent_env,
        "agent_name": args.agent_name,
        "max_papers": None,
        "max_duration_per_task_in_hours": duration,
        "mode": mode,
        "docker_image": args.docker_image,
        "dockerfile_name": "ExpDockerfile_default",
        "eval_gen_prompt": "evaluation/prompts/eval_gen_prompt.txt",
        "eval_judge_prompt": "evaluation/prompts/eval_judge_prompt.txt",
        "eval_judge_setup_prompt": "evaluation/prompts/eval_judge_setup_prompt.txt",
        "eval_judge_setup_partial_prompt": "evaluation/prompts/eval_judge_setup_partial_prompt.txt",
        "eval_judge_setup_monitor_prompt": "evaluation/prompts/eval_judge_setup_monitor_prompt.txt",
        "do_exec_check": not args.no_exec_check,
        "llm_judge_config_filename": args.judge_env or "",
        "judge_agent_name": "",
        # Fields added by evaluation/parallel_eval.py:create_main_task_config
        "paper_id": int(paper_id),
        "input_paper_tasks_folder": str(Path(tasks_file).parent),
        "input_paper_tasks_filename": tasks_file,
        "output_folder": output_folder,
        "output_log_folder": output_log_folder,
        "output_eval_gen_filename": f"{output_folder}/{paper_id}_eval_gen.json",
        "output_eval_judge_filename": f"{output_folder}/{paper_id}_eval_judge.json",
        "pdf_url": details.get("pdf_url"),
        "github_url": rec0["github_url"],
        # Upstream "specific_tasks": [paper_id, task_index, duration]
        "specific_tasks": [[paper_id, idx, duration] for idx in indices],
        # Harbor parity settings (see harbor_parity/README.md)
        "iteration_list": iterations,
        "pinned_commit": rec0["commit"],
        "pinned_repo": rec0["repo"],
        "pinned_setup_scripts_filename": scripts_file,
        "setup_apis_source": setup_apis_dir,
        "exclude_setup_apis_from_diff": True,
        "prune_docker_after_paper": not args.no_prune,
        "harbor_parity": manifest,
    }
    if args.openhands_python:
        config["openhands_python"] = args.openhands_python
    if args.dry_run:
        config["dry_run_stop_before_agent"] = True
    return config


def attempt_marker(args, model_id: str, conference: str, paper_id: str, iteration: int) -> Path:
    return (EXP_BENCH_DIR / PARITY_DIR / "attempts" / args.mode / f"{args.agent_name}_{model_id}"
            / f"{conference}_{paper_id}_iter_{iteration}_duration_{args.max_duration}.json")


# --------------------------------------------------------------------------- main


def docker_image_info(image: str) -> str | None:
    probe = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}} {{.Architecture}}"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return None
    commit = subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", image, "/helper_openhands/OPENHANDS_COMMIT"],
                            capture_output=True, text=True)
    return f"{probe.stdout.strip()} | {commit.stdout.strip() or 'OpenHands commit not recorded (image built from upstream Dockerfile?)'}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["generate", "judge"])
    parser.add_argument("--task-ids", nargs="+", default=[str(DEFAULT_TASK_IDS)],
                        help="Files with one task id per line and/or literal task ids.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="Harbor adapter task bundle (.json or .json.gz).")
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE_DIR,
                        help="Cache of ground-truth sources downloaded at the pinned commits (shared with the adapter).")
    parser.add_argument("--agent-env", required=True,
                        help='Agent LLM env file (export MODEL="..." plus credentials), relative to benchmark/exp_bench.')
    parser.add_argument("--judge-env", help="Judge env file (MODEL, OPENAI_API_KEY, OPENAI_BASE_URL); judge mode only.")
    parser.add_argument("--iterations", type=int, default=1, help="Run iterations 1..N (upstream 'iterations').")
    parser.add_argument("--iteration", type=int, nargs="+", help="Run only these iteration numbers (e.g. --iteration 2).")
    parser.add_argument("--max-duration", type=float, default=0.5,
                        help="Agent time limit per task in hours (upstream run_parallel_gen_evals default 0.5).")
    parser.add_argument("--parallel", type=int, default=6, help="Tasks of one paper run in parallel (parallelization_factor).")
    parser.add_argument("--agent-name", default="openhands")
    parser.add_argument("--docker-image", default="exp-bench-image")
    parser.add_argument("--output-root", default="outputs/evaluation")
    parser.add_argument("--openhands-python", help="Override the OpenHands interpreter path inside the image.")
    parser.add_argument("--no-exec-check", action="store_true", help="Judge without the execution check (E = 0).")
    parser.add_argument("--no-prune", action="store_true",
                        help="Do not run upstream's docker container/image/volume/builder prune after each paper.")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate only: clone, mask and build the prompt in the container, then stop before the agent.")
    parser.add_argument("--plan-only", action="store_true", help="Write the per-paper configs and exit.")
    parser.add_argument("--retry", action="store_true",
                        help="Re-launch (paper, iteration) pairs that were already attempted but have missing outputs "
                             "(e.g. agent timeouts, which upstream leaves unjudged). Off by default so each task gets "
                             "exactly one attempt per iteration.")
    args = parser.parse_args()

    os.chdir(EXP_BENCH_DIR)
    if args.mode == "judge" and not args.judge_env:
        sys.exit("judge mode needs --judge-env")
    if args.dry_run and args.mode != "generate":
        sys.exit("--dry-run is only for generate")
    for env_file in filter(None, [args.agent_env, args.judge_env]):
        if not Path(env_file).is_file():
            sys.exit(f"{env_file} not found")
        # The file is sourced inside the upstream container (working directory /), where only
        # these benchmark/exp_bench directories are mounted (main_eval.py:run_docker_container).
        if Path(env_file).is_absolute() or Path(env_file).parts[0] not in MOUNTED_DIRS:
            sys.exit(f"{env_file} must be a relative path inside one of {sorted(MOUNTED_DIRS)} "
                     "(e.g. evaluation/setup/...), because it is read inside the container")
    if args.judge_env:
        judge_env = read_env_file(Path(args.judge_env))
        if not judge_env.get("MODEL") or not judge_env.get("OPENAI_API_KEY"):
            sys.exit(f"{args.judge_env} must export MODEL (judge model, no provider prefix) and OPENAI_API_KEY")

    bundle, bundle_sha = load_bundle(args.bundle)
    by_id = {t["task_id"]: t for t in bundle["tasks"]}
    task_ids = read_task_ids(args.task_ids)
    unknown = [t for t in task_ids if t not in by_id]
    if unknown:
        sys.exit(f"Task ids not in the bundle: {unknown}")
    iterations = sorted(set(args.iteration)) if args.iteration else list(range(1, args.iterations + 1))
    if args.dry_run:
        iterations = [0]  # keeps dry-run files apart from real iterations 1..N
    model_id = model_identifier(Path(args.agent_env))

    image = docker_image_info(args.docker_image) if not args.plan_only else "not checked (--plan-only)"
    if image is None:
        sys.exit(
            f"Docker image {args.docker_image} not found. Build it first (see harbor_parity/README.md):\n"
            "  docker build --build-arg BASE_IMAGE=continuumio/miniconda3:25.1.1-2 --build-arg AZURE_ORG_PATCH=0 "
            f"-t {args.docker_image} -f ExpDockerfile_default ."
        )
    fork_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "bundle": str(args.bundle), "bundle_sha256": bundle_sha,
        "task_ids": task_ids, "iterations": iterations, "model_id": model_id,
        "docker_image": f"{args.docker_image}: {image}", "fork_commit": fork_commit,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    log(f"{len(task_ids)} tasks, iterations {iterations}, model id {model_id}")
    log(f"bundle sha256 {bundle_sha}")
    log(f"image {manifest['docker_image']}")

    papers: dict[tuple[str, str], list[int]] = defaultdict(list)
    for task_id in task_ids:
        rec = by_id[task_id]
        papers[(rec["conference"], rec["paper_id"])].append(rec["task_index"])
    all_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in bundle["tasks"]:
        all_records[(rec["conference"], rec["paper_id"])].append(rec)

    setup_apis_dir = write_setup_apis_dir()
    config_dir = Path("outputs/evaluation/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PARITY_DIR / f"manifest_{args.mode}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    failures = []
    for (conference, paper_id), indices in papers.items():
        indices = sorted(indices)
        tasks_file, scripts_file = write_paper_files(conference, paper_id, all_records[(conference, paper_id)],
                                                     indices, args.source_cache)
        rec0 = by_id[f"{conference}-{paper_id}-{indices[0]}"]
        output_folder, _ = output_folders(args, conference, paper_id, model_id)
        todo = [it for it in iterations
                if args.dry_run or not all(is_done(output_folder, paper_id, i, it, args.max_duration, args.mode) for i in indices)]
        if args.mode == "judge":
            # Only judge iterations that have generation outputs (upstream judge.py skips the rest).
            todo = [it for it in todo
                    if any(eval_file(output_folder, paper_id, i, it, args.max_duration, "generate").exists() for i in indices)]
        attempted = [it for it in todo if attempt_marker(args, model_id, conference, paper_id, it).exists()]
        if attempted and not args.retry and not args.dry_run:
            log(f"{conference}/{paper_id}: iterations {attempted} were already attempted and have missing outputs; "
                "not re-running them (pass --retry to re-attempt)")
            todo = [it for it in todo if it not in attempted]
        if not todo:
            log(f"{conference}/{paper_id} tasks {indices}: nothing to {args.mode} for iterations {iterations}")
            continue
        config = build_config(args, args.mode, conference, paper_id, rec0, indices, todo, model_id,
                              (tasks_file, scripts_file, setup_apis_dir), manifest)
        config_path = config_dir / f"harbor_parity_{conference}_{paper_id}_{args.mode}{'_dry_run' if args.dry_run else ''}_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        log(f"{conference}/{paper_id} tasks {indices} iterations {todo} @ {rec0['commit'][:12]} -> {config_path}")
        if args.plan_only:
            continue
        if not args.dry_run:
            for it in todo:
                marker = attempt_marker(args, model_id, conference, paper_id, it)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"task_indices": indices, "launched": datetime.now().isoformat(timespec="seconds"),
                                              "config": str(config_path)}))
        env = dict(os.environ, PYTHONPATH=str(EXP_BENCH_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run([sys.executable, "evaluation/main_eval.py", f"--task_config={config_path}"], env=env)
        if result.returncode != 0:
            log(f"ERROR: main_eval.py failed for {conference}/{paper_id} (exit {result.returncode}); continuing")
            failures.append(f"{conference}/{paper_id}")
    log(f"done; manifest {manifest_path}" + (f"; failed papers: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
