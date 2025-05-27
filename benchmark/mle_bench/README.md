# Running MLE Bench with Curie

## Directory Structure
The `mle_bench` directory contains several benchmark tasks from MLE Bench:

- [`aptos2019-blindness-detection/`](./aptos2019-blindness-detection/): Diabetic retinopathy detection task
- [`histopathologic-cancer-detection/`](./histopathologic-cancer-detection/): Cancer detection from histopathology images
- [`siim-isic-melanoma-classification/`](./siim-isic-melanoma-classification/): Melanoma classification task
- [`dog-breed-identification/`](./dog-breed-identification/): Dog breed classification task
- More incoming...


Each task directory contains:
- A question file used to ask Curie (e.g., [aptos2019-blindness-detection-question.txt](aptos2019-blindness-detection/aptos2019-blindness-detection-question.txt))
- Curie's auto-generated experiment report (e.g. [aptos2019-blindness-detection_report.md](aptos2019-blindness-detection/aptos2019-blindness-detection_20250522225727_iter1.md))
- Curie's experimentation logs

## Setup and Installation

### 1. Install MLE Benchmark
- Setup kaggle credential `~/.kaggle/kaggle.json`
- Install correct sqlite version (to fix the bug in mle-bench)
 
```bash
conda create --name sqlite3-49-0 python=3.11
conda activate sqlite3-49-0
conda install sqlite=3.49
```

- Install `mlebench`

```bash
git clone https://github.com/openai/mle-bench.git
cd mle-bench
git lfs fetch --all
git lfs pull
pip install -e .
```

### 2. Download Dataset
Run `mlebench prepare` with the specific task ID:
```bash
conda activate sqlite3-49-0  
mlebench prepare -c <task-id>  # e.g., dog-breed-identification
```
The data will be saved to `$HOME/.cache/mle-bench/data`.

## Running Curie on MLE Bench Tasks

### 1. Select a Task
Choose one of the available tasks from the MLE-Bench (E.g. `siim-isic-melanoma-classification`).

### 2. Run Curie
Use the following command format:
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/<task-dir>/<task>-question.txt --dataset_dir <abd_path_to_dataset> --task_config curie/configs/mle_config.json 
```
<!-- 
Example for dog breed identification:
```bash
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt --task_config curie/configs/mle_config.json 
``` -->

<!-- ### 3. Grade Your Submission
After running Curie, grade your submission using:
```bash
conda activate sqlite3-49-0  
mlebench grade-sample your_submission.csv <task-id>
``` -->

## Additional Information

### Question Generation
MLE Bench provides `description.md` for each problem. We use the following prompt to convert the description into a research question:
```
Convert this Kaggle competition into a question to the ai agent (be concise): introduce the problem, goal, and all necessary details to guide the agent to find the best performing model/configuration: 
<description.md>
```


 
<!-- docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/amberljc/dev/Curie/curie:/curie:ro -v /home/amberljc/dev/Curie/benchmark:/benchmark:ro -v /home/amberljc/dev/Curie/logs:/logs -v /home/amberljc/dev/Curie/starter_file:/starter_file:ro -v /home/amberljc/dev/Curie/workspace:/workspace -v /:/all:ro --network=host -d --name exp-test exp-agent-image -->
 