# Harbor parity runs

This branch runs the EXP-Bench harness on the same tasks as the
[Harbor EXP-Bench adapter](https://github.com/harbor-index/adapters/pull/28), so scores
from both can be compared. Without the keys `run_parity.py` adds to the task config, the
harness behaves as upstream.

## Changes to the harness

| Change | Where | Why |
|---|---|---|
| Claude Code agent (`agent_name: claude-code`) | `helper/utils.py:call_claude_code_with_prompt`, `evaluation/eval.py` | Upstream's OpenHands build (Feb 2025) cannot call current models. Same install procedure as Harbor's `claude-code` agent (installs whichever of curl, bash, nodejs, npm and procps the image lacks, then the pinned release), same environment and flags, in a sandbox container with the repository at `/workspace`, as in the OpenHands sandbox. |
| Pinned repository commits | `evaluation/eval.py:git_clone_repo` | Upstream clones `HEAD`; the adapter pins each repository to the commit before its tasks were built. |
| Ground-truth sources at the pinned commit | `evaluation/judge.py`, `helper/utils.py:concatenate_setup_scripts` | Upstream downloads them from `main`/`master` at judge time. |
| OpenAI-compatible judge endpoint, Claude judges | `model.py` | Upstream supports Azure only. Claude models need a non-empty user turn, a strict JSON schema and no temperature; prompts are unchanged. |
| Monitor log truncated to the judge context; patch chunks leave room for the reference files | `evaluation/judge.py`, `helper/utils.py` | Upstream fails with a context-length error on long logs or when large reference files are added to full-size chunks; same rules as the adapter. |
| Build arguments for version pins | `ExpDockerfile_default`, `build_image.sh` | The floating versions upstream installs no longer build the image. |
| Selected iterations, optional docker prune, `docker exec -i` | `evaluation/main_eval.py` | Run 2 and 3 separately; keep other containers on shared hosts; run without a TTY. |

Task list and records: `data/parity_task_ids.txt` (the adapter's parity subset, 57 tasks) and
`data/exp_bench_tasks.json.gz` (the adapter's task bundle).

## Run

From `benchmark/exp_bench` on a Linux x86_64 host with Docker, Python ≥ 3.12 and NVIDIA GPUs
(each sandbox gets `--gpus 1`, as each parity task does on the Harbor side; `--sandbox-gpus`
changes it):

```bash
harbor_parity/build_image.sh
cp harbor_parity/env-parity-agent.sh.example evaluation/setup/env-parity-agent.sh   # fill in
cp harbor_parity/env-parity-judge.sh.example evaluation/setup/env-parity-judge.sh   # fill in

python harbor_parity/run_parity.py generate --agent-env evaluation/setup/env-parity-agent.sh --iteration 1 2 3
python harbor_parity/run_parity.py judge --agent-env evaluation/setup/env-parity-agent.sh \
    --judge-env evaluation/setup/env-parity-judge.sh --iteration 1 2 3
python harbor_parity/collect_results.py --agent-env evaluation/setup/env-parity-agent.sh \
    --iterations 1 2 3 --harbor-jobs <harbor job dir per run> --out outputs/harbor_parity/parity_results.json
```

Each task gets one attempt per iteration. Rerun rule, the same as for the Harbor jobs: a
run whose Claude Code stream ends with `success` or `error_max_turns`, or that hits the time
limit, is a result (0 without an answer); any other run (API error, `error_during_execution`,
no result event, setup failure) is an infrastructure failure. The launcher writes the status
next to the agent log (`status.txt`); `collect_results.py` excludes infrastructure failures,
lists them for a rerun (`--retry`) and lists scored runs without a judge result for regrading. Harbor side
(adapter repository):

```bash
harbor run -c src/exp-bench/run_exp-bench_parity.yaml
```
