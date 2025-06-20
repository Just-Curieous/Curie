#!/usr/bin/env python3
import os
import sys
import argparse
import json
import subprocess
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Curie Agent Docker Entry Point")
    parser.add_argument("--base_dir", required=True, help="Absolute path to base directory")
    parser.add_argument("--prompt_path", required=True, help="Path to prompt file (contains the question)")
    parser.add_argument("--code_repo_path", required=True, help="Path to the cloned GitHub repo")
    parser.add_argument("--max_timeout_in_seconds", type=int, default=3600, help="Timeout (unused currently)")

    args = parser.parse_args()

    # Read the full question from prompt file
    with open(args.prompt_path, "r") as f:
        question = f.read().strip()

    print("✅ Running Curie agent with the following parameters:")
    print(f"    question: {question[:100]}...")
    print(f"    repo: {args.code_repo_path}")
    print(f"    base_dir: {args.base_dir}")

    # Prepare temp config
    cfg = {
        "use_docker": False,
        "dataset_dir": "/workspace",
        "workspace_base": "/workspace",
        "dockerfile_name": "Dockerfile",
        "docker_image": "exp-agent-image",
        "model_name": "gpt-4o-mini"
    }
    tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp_cfg)
    tmp_cfg.close()

    # Make sure curie/ is in path
    sys.path.append(os.path.dirname(__file__))

    # Launch Curie agent (assuming main is implemented in curie.main)
    command = [
        sys.executable, "-m", "curie.main",
        "--question", question,
        "--workspace_name", args.code_repo_path,
        "--iterations", "1",
        "--no_docker",
        "--task_config", tmp_cfg.name
    ]

    print(f"🚀 Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
