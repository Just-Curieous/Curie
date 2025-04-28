
## Installation

- install correct sqlite
 
```bash
conda create --name sqlite3-49-0 python=3.11
conda activate sqlite3-49-0
conda install sqlite=3.49
```

- setup kaggle
- mlebench

```bash
git lfs fetch --all
git lfs pull
pip install -e .
```

## dataset
```bash
conda activate sqlite3-49-0  
mlebench prepare -c dog-breed-identification
mlebench prepare -c cassava-leaf-disease-classification 
```
The data will be saved to `/home/ubuntu/.cache/mle-bench/data`

## Run Curie
```bash
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt   --task_config curie/configs/mle_dog_config.json  --report
```


## Grade submission

```
conda activate sqlite3-49-0  

mlebench grade-sample  workspace/dog-breed-identification_b3307073-883d-4804-b6da-15ac6f746599/submission_control_group_partition_1.csv dog-breed-identification 
```


## prompt to generate question
```
Convert this Kaggle competition into a question to the ai agent (be concise): introduce the problem, goal, and all necessary details to guide the agent to find the best performing model/configuration:
```



## for testing

- sucessful run:
```
dog-breed-identification_20250427163751_iter1
```

```
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/amberljc/dev/Curie/curie:/curie:ro -v /home/amberljc/dev/Curie/benchmark:/benchmark:ro -v /home/amberljc/dev/Curie/logs:/logs -v /home/amberljc/dev/Curie/starter_file:/starter_file:ro -v /home/amberljc/dev/Curie/workspace:/workspace -v /:/all:ro --network=host -d --name exp-test exp-agent-image

```