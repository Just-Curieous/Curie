import os
import re
import json
import subprocess
from pathlib import Path

# Configuration
LOGS_DIR = "../../logs/mdgen_task_index_1_iter_1_duration_0.67_20250725153116_iter1"
OUTPUT_BASE = "outputs/evaluation/neurips2024/93022/curie/test"

# Standard output name
def get_standard_output_name(md_filename):
    m = re.match(r".*?(\d+)_task_index_(\d+)_iter_(\d+)_duration_([0-9.]+)", md_filename)
    if m:
        paper_id, task_index, iter_num, duration = m.groups()
        print(f"[DEBUG] File: {md_filename} -> paper_id: {paper_id}, task_index: {task_index}, iter: {iter_num}, duration: {duration}")
        return f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.json", \
               f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.patch"
    else:
        print(f"[DEBUG] File: {md_filename} -> Failed to match paper_id/task_index/iter/duration")
        base = os.path.splitext(md_filename)[0]
        return base + ".json", base + ".patch"

# Only iterate through .md files
for file in os.listdir(LOGS_DIR):
    file_path = os.path.join(LOGS_DIR, file)
    if not os.path.isfile(file_path):
        continue
    if not file.endswith(".md"):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract Experiment Design
    design_match = re.search(
        r"### Experiment Design\s*([\s\S]*?)(^### |^## |\Z)", content, re.MULTILINE)
    design = design_match.group(1).strip() if design_match else ""

    # Extract Conclusion and Future Work
    conclusion_match = re.search(
        r"## Conclusion and Future Work\s*([\s\S]*?)(^## |\Z)", content, re.MULTILINE)
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

    json_name, patch_name = get_standard_output_name(file)
    out_json_path = os.path.join(OUTPUT_BASE, json_name)
    out_patch_path = os.path.join(OUTPUT_BASE, patch_name)

    if design and conclusion:
        result = {"design": design, "conclusion": conclusion}
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Extracted: {file_path} -> {out_json_path}")
        # Generate patch file (assuming workspace_name field or inferrable repo path)
        # This is just an example: assuming repo path is ../../workspace/mdgen_eval_task_index_1_iter_1_duration_0.67
        repo_dir = "../../workspace/mdgen_eval_task_index_1_iter_1_duration_0.67"
        # Generate patch (diff current repo with HEAD)
        try:
            with open(out_patch_path, "w", encoding="utf-8") as pf:
                subprocess.run(["git", "-C", repo_dir, "diff"], stdout=pf, check=True)
            print(f"Generated patch: {out_patch_path}")
        except Exception as e:
            print(f"Failed to generate patch: {e}")
    else:
        print(f"Design/conclusion not found: {file_path}") 