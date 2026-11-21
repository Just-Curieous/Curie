# EXP-Bench parity harness: original side

This branch (`harbor-parity`) prepares the **original** EXP-Bench harness
(`benchmark/exp_bench`, generate → judge) for parity experiments against the Harbor
adapter (`harbor-framework/adapters`, `src/exp-bench`). It runs the same tasks under the
same conditions as the adapter while keeping upstream's agent prompt, masking, judge
prompts and score definitions unchanged.

Everything is opt-in: without the new config keys the upstream code paths behave as
before. The changes are limited to task selection, repository pinning, the judge's API
client, OpenHands version pinning and bookkeeping.

## What changed

| Area | File | Change | Why |
|---|---|---|---|
| Task selection | `harbor_parity/run_parity.py` (new) | Runs an explicit list of task ids `<conference>-<paper_id>-<task_index>` (`task_index` = position in `<paper_id>_complete_final.json`). Writes each paper's question list and pinned ground-truth sources from the task bundle, builds upstream per-paper configs (with upstream `specific_tasks`) and calls the unmodified `evaluation/main_eval.py`, one paper after another as `parallel_eval.py` does. | Same tasks and task records on both sides. |
| Task data | `harbor_parity/data/exp_bench_tasks.json.gz` (new) | The Harbor adapter's task bundle, byte-identical copy of its `src/exp_bench/data/exp_bench_tasks.json.gz` (461 records; sha256 of the uncompressed JSON `2797cdf7ca3f1ea8bad984a9db74880f42f57ed178f32dfc3731127a4ba5206b`, printed by the runner and stored in each manifest). Its `task_data` equals upstream's `outputs/logs/<conf>/<paper>/<paper>_complete_final.json` for all 51 papers; the runner re-checks this and warns on any difference. Ground-truth sources are downloaded from `raw.githubusercontent.com/<repo>/<pinned commit>/…` and checked against the bundle's `setup_script_sha256`, exactly as the adapter does (same cache, `~/.cache/harbor-exp-bench-adapter`). | One source of truth for both sides. |
| Parity subset | `harbor_parity/data/parity_task_ids.txt` (new) | Copy of the adapter's provisional subset (33 tasks, 19 papers). | |
| Repository pinning | `evaluation/eval.py` (`git_clone_repo`, `update_submodules`, `process_inputs`) | With `pinned_commit` in the config: `git init` + `git fetch --depth 1 origin <commit>` + `checkout FETCH_HEAD`, submodules with `--depth 1` (falling back to a full update), exactly as the adapter's `prepare_workspace.sh`. Without it: upstream `git clone` of `HEAD`. | Upstream clones the current `HEAD`, which drifts. |
| Agent credentials folder | `evaluation/eval.py`, `run_parity.py` | `setup_apis_exp/` gets `env.sh` exporting `OPENAI_API_KEY`/`OPENAI_BASE_URL` from `EXP_BENCH_TASK_OPENAI_API_KEY`/`EXP_BENCH_TASK_OPENAI_BASE_URL` (empty unless set) and is added to `.git/info/exclude`. | Same as the adapter. Upstream copies its whole `evaluation/setup/` (including the agent and judge API keys) into the repo, where the files also land in the judged diff. |
| Judge client | `model.py` | `OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or OPENAI_API_BASE)` when `OPENAI_API_KEY` is set; upstream `AzureOpenAI` otherwise. Model from `MODEL` as before. Transport-level retries (connection errors, timeouts, 429, 5xx) with the adapter's policy; a `None` message content counts as unparsable (default output, then upstream's retry). | Same endpoint, model and retry policy as the adapter's judge. Request body checked identical: `model`, `messages` = [system prompt, empty user turn], `stop: null`, `response_format: json_object`, temperature only for non-o1/o3 models. |
| Judge ground truth | `evaluation/judge.py`, `helper/utils.py` (`concatenate_setup_scripts`) | With `pinned_setup_scripts_filename`: `setup_ground_truth_scripts` come from the bundle (files at the pinned commit), formatted as upstream. Without it: upstream download from `main`/`master`. `judge.py` also honours `specific_tasks`. | Same text as the adapter's judge. |
| OpenHands version | `ExpDockerfile_default`, `harbor_parity/patch_openhands.py` (new) | `OPENHANDS_REF` build arg (default: upstream's `02bc7de36dda10d890d39ac05fee16a0d55193f6`); upstream's two line-number `sed` patches (docker.sock mount, sandbox uid 0 → 123) are applied by content anchors, byte-identical to the `sed` result at `02bc7de` and failing loudly on other versions if an anchor is gone; upstream's Azure-organization patches of litellm are behind `AZURE_ORG_PATCH` (default 1 = upstream). The commit actually built is recorded in `/helper_openhands/OPENHANDS_COMMIT`. | So the OpenHands version can match the Harbor side. |
| Build reproducibility | `ExpDockerfile_default`, `harbor_parity/build_image.sh` (new) | Build args `BASE_IMAGE`, `NPM_VERSION`, `CONDA_NODEJS_SPEC`, `CONDA_POETRY_SPEC` (defaults = upstream's floating versions). `build_image.sh` pins early-2025 releases. | Upstream's `npm install -g npm@latest` now fails (npm 12 needs Node ≥ 22). |
| Agent LLM endpoint | `helper/utils.py` | Unknown models get cost 0 instead of a `KeyError` (cost is bookkeeping only); optional `OH_<SECTION>__<KEY>` env entries are written to OpenHands' `config.toml`; optional `openhands_python` config key. The endpoint itself was already configurable (`*_API_KEY` → `api_key`, `*_API_BASE` → `base_url`). | OpenAI/Anthropic-compatible base URL such as `https://app.ppapi.ai/v1`. |
| Runner robustness | `evaluation/main_eval.py` | `docker exec -it` only when stdin is a terminal (`-i` otherwise); optional `iteration_list`; `prune_docker_after_paper: false` (`--no-prune`) limits the post-paper cleanup to OpenHands sandboxes. | Background runs; running one iteration at a time; shared Docker hosts. |
| Dry run | `evaluation/eval.py` | `dry_run_stop_before_agent` stops each task after clone, masking and prompt generation and saves them to `*_dry_run.json` (iteration 0). | Check the pipeline without an LLM. |
| Results | `harbor_parity/collect_results.py` (new) | Converts `*_eval_judge.json` to the adapter's `reward.json` values and compares with Harbor job directories. | See "Results". |
| Smoke test | `harbor_parity/stub_llm_server.py` (new) | OpenAI-compatible stub that records requests and returns canned judge JSON or an immediate `finish` tool call. | Zero-cost end-to-end test. |

Not changed: `evaluation/prompts/*` (identical to the adapter's copies), masking
(`mask_repo`), `generate_patch`, the judge pipeline order, chunking, monitor, execution
check, and how scores are written.

## Reproduction

All commands run from `benchmark/exp_bench` on a Linux x86_64 host with Docker and
Python ≥ 3.12 (the upstream scripts use 3.12 f-string syntax; the host side needs only
the standard library). A GPU is optional; upstream passes `--gpus all` when
`nvidia-smi` works.

### 1. Build the image

```bash
harbor_parity/build_image.sh
```

The script prints the OpenHands commit it built. On Apple silicon add
`--platform linux/amd64` (upstream's `environment.yml` pins linux-64 packages). To build a
different OpenHands version, e.g. to match the Harbor side:

```bash
OPENHANDS_REF=<tag-or-commit> harbor_parity/build_image.sh
```

### 2. Credentials

```bash
mkdir -p evaluation/setup
cp harbor_parity/env-parity-agent.sh.example evaluation/setup/env-parity-agent.sh
cp harbor_parity/env-parity-judge.sh.example evaluation/setup/env-parity-judge.sh
# edit both files; evaluation/setup/ is gitignored
```

- Agent: `MODEL` is the litellm model string, the same string as `harbor run -m`, with
  `OPENAI_API_KEY` + `OPENAI_API_BASE` (or the `ANTHROPIC_*` pair). Keep
  `OH_LLM__NATIVE_TOOL_CALLING="true"` from the template: Harbor's `openhands` agent always
  sets `LLM_NATIVE_TOOL_CALLING=true`. Even so, OpenHands 0.27 only uses native tool calls
  when the litellm version locked at `02bc7de` (early 2025) knows the model
  (`litellm.supports_function_calling`); for unknown or newer model names it falls back to
  prompt-based tools. Choose a parity model that 0.27 handles natively, e.g.
  `openai/gpt-4o-mini`, `openai/o3-mini` or `anthropic/claude-3-5-haiku-20241022`, or move
  both sides to a newer OpenHands.
- Judge: `MODEL` is the judge model as the endpoint names it, without a provider
  prefix, and must equal `EXP_BENCH_JUDGE_MODEL` on the Harbor side (upstream: `o3-mini`).
- Optional, for experiments that call an LLM themselves: export
  `EXP_BENCH_TASK_OPENAI_API_KEY` / `EXP_BENCH_TASK_OPENAI_BASE_URL` in the shell that
  runs `run_parity.py` (written to `setup_apis_exp/env.sh`, like the adapter).

### 3. Dry run (no LLM calls)

```bash
python harbor_parity/run_parity.py generate --dry-run --no-prune \
  --task-ids neurips2024-98316-5 --agent-env evaluation/setup/env-parity-agent.sh
```

This starts the upstream container, clones `py-why/causal-learn` at `9689c1bd`, masks
`causallearn/search/FCMBased/ANM/ANM.py` and the README, builds the agent prompt and
stops. Inspect
`outputs/evaluation/logs/neurips2024/98316/openhands/<model-id>/98316_task_index_5_iter_0_duration_0.5_eval_gen_openhands_dry_run.json`.

### 4. Optional smoke test with a stub endpoint (no paid calls)

Checked on 2026-11-21 (Apple silicon, image built with `--platform linux/amd64`): the
generate run below made one LLM call, OpenHands finished, `no_answer` and an empty patch
were recorded; the judge run sent the monitor, design/conclusion and setup prompts (with
the pinned, hash-checked ground-truth source), ran the execution check in a fresh
container, and `collect_results.py` produced D 50 / I 25 / C 0 / E 0 / monitor_pass 100
from the stub's fixed answers. With the stub's model name `openai/stub-model` OpenHands
uses prompt-based tools; with `MODEL="openai/gpt-4o-mini"` (and
`--output-root outputs/harbor_parity/stub_evaluation` to keep it apart) it sends native
tool definitions.

```bash
python harbor_parity/stub_llm_server.py --host 127.0.0.1 --port 8765 &
python harbor_parity/run_parity.py generate --no-prune --iteration 9 \
  --task-ids neurips2024-98316-5 --agent-env evaluation/setup/env-stub-agent.sh
python harbor_parity/run_parity.py judge --no-prune --iteration 9 \
  --task-ids neurips2024-98316-5 --agent-env evaluation/setup/env-stub-agent.sh \
  --judge-env evaluation/setup/env-stub-judge.sh
```

after copying `harbor_parity/env-stub-agent.sh.example` and
`harbor_parity/env-stub-judge.sh.example` to `evaluation/setup/env-stub-agent.sh` and
`evaluation/setup/env-stub-judge.sh` (they point at `http://host.docker.internal:8765/v1`;
on Linux, where the upstream container uses host networking, use
`http://127.0.0.1:8765/v1`). OpenHands makes one LLM call and finishes; the judge runs the monitor,
design/conclusion and setup prompts and the execution check. Requests are saved under
`outputs/harbor_parity/stub_requests/`.

### 5. Parity runs

One iteration = one run over the subset. Generate, then judge, per iteration:

```bash
for i in 1 2 3; do
  python harbor_parity/run_parity.py generate --iteration $i --agent-env evaluation/setup/env-parity-agent.sh
  python harbor_parity/run_parity.py judge --iteration $i --agent-env evaluation/setup/env-parity-agent.sh \
    --judge-env evaluation/setup/env-parity-judge.sh
done
```

- Papers run one after another; up to `--parallel` (default 6) tasks of a paper run at
  once. Expect roughly (number of papers) × (time limit + clone/teardown) per iteration,
  about 10–12 hours for the 19-paper subset at the default 0.5 h limit.
- Each (paper, iteration) is launched once. Re-running the command skips completed
  work and does **not** re-attempt tasks that have no output (upstream writes nothing
  for a task whose agent hit the time limit); pass `--retry` to re-attempt them, e.g.
  after a machine crash.
- Defaults match upstream `run_parallel_gen_evals.py`: `--max-duration 0.5` hours,
  OpenHands `--max-iterations 30`, judge with execution check (`--no-exec-check` to
  skip). Use `--no-prune` on machines shared with other Docker work: upstream otherwise
  runs `docker container/image/volume/builder prune -f` after every paper.
- Each invocation writes `outputs/harbor_parity/manifest_<mode>_<time>.json` (bundle
  hash, task ids, iterations, image id and OpenHands commit, fork commit).

Outputs keep upstream's layout:
`outputs/evaluation/<conf>/<paper_id>/openhands/<model-id>/<paper_id>_task_index_<i>_iter_<k>_duration_0.5_eval_{gen,judge}.json`
plus the patch, config and logs.

### 6. Harbor side (for reference)

From the adapters repository (see `src/exp-bench/README.md` there):

```bash
cd src/exp-bench && uv run exp-bench --split parity --output-dir ../../datasets/exp-bench-parity && cd ../..
harbor run -c src/exp-bench/run_exp-bench_parity.yaml   # openhands, same model, same OpenHands version
```

with the judge env `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `EXP_BENCH_JUDGE_MODEL` equal to
this side's judge `MODEL`, agent kwargs `max_iterations: 30`, and the adapter's default
0.5 h agent timeout.

### 7. Results

```bash
python harbor_parity/collect_results.py --agent-env evaluation/setup/env-parity-agent.sh \
  --iterations 1 2 3 --harbor-jobs <jobs>/run1 <jobs>/run2 <jobs>/run3 \
  --out outputs/harbor_parity/parity_results.json
```

Per task and run the upstream judge output is converted with the adapter's definitions
(`design` = `design_score`/100, `implementation` = `setup_score`/100, both clamped to
[0, 100] and 0 when unparsable; `conclusion` = `conclusion_score == "correct"`;
`execution` = `execution_success`; the conjunctive metrics; `monitor_pass` = 0 when the
log monitor flagged the run). The script prints, per metric, the per-run means, mean ±
sample SEM across runs (ddof = 1), whether the original and Harbor run ranges overlap, and
the per-task MSE (mean over tasks of the squared difference between the per-task means of
the two sides) with its RMSE in percentage points. Tasks without a judge output (or Harbor
trials without rewards) count as 0 and are listed; `--missing exclude` drops them instead.
A Harbor job holding several attempts per task is split into one run per attempt.

## Known differences that remain

These come from how the two harnesses run the agent, not from the benchmark logic; they
should be stated in the parity report.

1. **OpenHands runtime.** Upstream runs OpenHands in its container with the Docker
   runtime: commands execute in an OpenHands sandbox image with the repository mounted
   at `/workspace`. Harbor's `openhands` agent installs OpenHands with pip into the task
   container and uses `RUNTIME=local`, so commands run in the adapter's
   `continuumio/miniconda3:25.1.1-2` image. Installed tools and Python differ.
2. **OpenHands version.** Upstream's `02bc7de` is release 0.27.0 plus 7 later commits
   (among them the CodeAct `think` tool), so `openhands-ai==0.27.0` from PyPI is not the
   same code. Harbor can install the exact commit with the agent kwarg
   `git_version: 02bc7de36dda10d890d39ac05fee16a0d55193f6` (and `python_version: "3.12"`);
   alternatively build this side with `OPENHANDS_REF=<version Harbor installs>`. Two
   caveats: the 0.27-era local runtime runs `poetry` against an OpenHands source checkout,
   which a pip install may not satisfy; and a pip/uv install from git resolves dependencies
   afresh (e.g. a newer litellm) while this image uses OpenHands' `poetry.lock`. Check both
   on the Harbor side before choosing the version.
3. **Agent options.** Harbor's agent sets `LLM_NATIVE_TOOL_CALLING=true` (mirrored by the
   template), `AGENT_ENABLE_PROMPT_EXTENSIONS=false`, `AGENT_ENABLE_BROWSING=false` and
   `SANDBOX_ENABLE_AUTO_LINT=true`. Upstream runs with prompt extensions on (microagents
   trigger, e.g. the `github` one on the word "git") and browsing on (`browser` and
   `web_read` tools are offered); at 0.27 the browsing field is `codeact_enable_browsing`, so
   Harbor's `AGENT_ENABLE_BROWSING` does not reach it. Align with `OH_AGENT__...` /
   `OH_SANDBOX__...` entries once the version is fixed. Upstream also runs with
   `--max-iterations 30` and `SANDBOX_TIMEOUT=600`; pass `max_iterations: 30` (and
   `OPENHANDS_SANDBOX_TIMEOUT=600` if supported) on the Harbor side.
4. **Agent timeouts.** Upstream raises when the agent exceeds the time limit and writes
   no output, so the task is never judged (counted as 0 here). Harbor grades timed-out
   trials. `collect_results.py --harbor-agent-timeout zero` scores Harbor timeouts as 0
   for a like-for-like comparison.
5. **Monitor input.** Upstream gives the log monitor the OpenHands console log; the
   adapter gives it all files in `/logs/agent` (truncated to fit).
6. **Output file name.** The prompt's answer path encodes the iteration
   (`..._iter_<k>_duration_0.5_eval_gen.json`); the adapter always writes `iter_1`, so the
   prompts are identical for iteration 1 and differ only in that digit for iterations 2–3.
7. **Execution check.** Upstream runs `reproduce_exp_bench.sh` in a fresh container of
   this image with the workspace mounted; the adapter runs it in the agent's container.

## Files

```
harbor_parity/
├── README.md                    # this file
├── run_parity.py                # task-id runner (generate / judge / dry run)
├── collect_results.py           # reward values, run statistics, MSE vs Harbor
├── build_image.sh               # pinned image build
├── patch_openhands.py           # anchored OpenHands patches (used by the Dockerfile)
├── stub_llm_server.py           # zero-cost smoke test endpoint
├── env-parity-agent.sh.example
├── env-parity-judge.sh.example
└── data/
    ├── exp_bench_tasks.json.gz  # Harbor adapter task bundle
    └── parity_task_ids.txt      # parity subset (provisional)
```
