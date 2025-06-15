import json, tempfile, os, sys, subprocess, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--prompt_path", required=True)
    p.add_argument("--code_repo_path", required=True)
    p.add_argument("--max_timeout_in_seconds", type=int)  # 目前不用
    args = p.parse_args()

    with open(args.prompt_path, "r") as f:
        question = f.read().strip()

    # -------- 写配置 --------
    cfg = {
        "use_docker":   False,
        "dataset_dir":  "/workspace",          # 现成目录即可
        "workspace_base": "/workspace",
        "dockerfile_name": "Dockerfile",
        "docker_image": "exp-agent-image", 
        "model_name":   "gpt-4o-mini"
    }
    tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp_cfg)
    tmp_cfg.close()

    # -------- 调 Curie --------
    sys.path.append(os.path.dirname(__file__))
    cmd = [
        sys.executable, "-m", "curie.main",
        "--question",       question,
        "--workspace_name", os.path.abspath(args.code_repo_path),
        "--iterations",     "1",
        "--no_docker",
        "--task_config",    tmp_cfg.name
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
