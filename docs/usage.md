# Usage

This section provides an overview of how to use the Curie project.

## Basic Usage - A simple question
- Algorithm performance analysis 

```bash
python3 -m curie.main -q "How does the choice of sorting algorithm impact runtime performance across different input distributions?" --task_config curie/configs/base_config.json
```

- ML configuration analysis
```bash
python3 -m curie.main -q "How does feature scaling (Min-Max vs. Standardization) impact the accuracy of a Logistic Regression model on the Iris dataset?" --task_config curie/configs/base_config.json
```

## Advanced Features - Work on your existing code base
 
### Tutorial for Reproducing 'Large Language Monkeys' Results

The paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) explores repeated sampling as a method to enhance reasoning performance in large language models (LLMs) by increasing inference compute. 

Download the related starter files under `workspace`.
```bash 
git submodule update --init --recursive 
```

As a LLM researcher, you are just curious about how does the number of repeatedly generated samples per question impact the overall success? (The concrete question can be found in our benchmark `benchmark/experimentation_bench/llm_reasoning/q1_simple_relation.txt`, which specify the location of corresponding starter files.)

```bash 
python3 -m curie.main --iterations 1 --question_file benchmark/experimentation_bench/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```

- You can check the logging under `logs/q1_simple_relation_<ID>.log`.

- You can check the reproducible experimentation process under `workspace/large_language_monkeys_<ID>`.


## Data Collection Notice

Curie collects anonymized questions to improve our research benchmark. No personal data is gathered. You can opt-out by setting:
```bash
export CURIE_TELEMETRY=false
```