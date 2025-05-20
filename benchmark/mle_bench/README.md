
## Instal [MLE Benchmark](https://github.com/openai/mle-bench)
- Setup kaggle credential `~/.kaggle/kaggle.json`.
- Install correct sqlite version (to fix the bug in mle-bench).
 
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

## Download Dataset
- Run `mlebench prepare -c <task-id>`, for example:
```bash
conda activate sqlite3-49-0  
mlebench prepare -c dog-breed-identification
```
The data will be saved to `$HOME/.cache/mle-bench/data`.

## Run Curie
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt --task_config curie/configs/mle_dog_config.json 
```

## Grade submission

```
conda activate sqlite3-49-0  
mlebench grade-sample  your_submission.csv dog-breed-identification 
```


## Prompt to generate question from the MLE Bench 
MLE Bench only provides `description.md` of the problem, we use this prompt to convert the description into a research question:
```
Convert this Kaggle competition into a question to the ai agent (be concise): introduce the problem, goal, and all necessary details to guide the agent to find the best performing model/configuration:
```


 
<!-- docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/amberljc/dev/Curie/curie:/curie:ro -v /home/amberljc/dev/Curie/benchmark:/benchmark:ro -v /home/amberljc/dev/Curie/logs:/logs -v /home/amberljc/dev/Curie/starter_file:/starter_file:ro -v /home/amberljc/dev/Curie/workspace:/workspace -v /:/all:ro --network=host -d --name exp-test exp-agent-image -->
 