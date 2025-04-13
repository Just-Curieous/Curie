
# Tutorial for Reproducing 'Large Language Monkeys' Results

The paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) explores repeated sampling as a method to enhance reasoning performance in large language models (LLMs) by increasing inference compute. 

1. Download the related starter files under `workspace`.
```bash
cd Curie
git submodule update --init --recursive 
```

2. Be curious.

As an LLM researcher, you are just curious how the number of repeatedly generated samples per question impacts the overall success. (The concrete question can be found in our benchmark `benchmark/experimentation_bench/llm_reasoning/q1_simple_relation.txt`, which specifies the location of corresponding starter files.)

```bash
cd Curie
python3 -m curie.main --iterations 1 --question_file benchmark/experimentation_bench/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```
(We pre-specify the starter file directory name inside `llm_reasoning_config.json`.)

- You can check the logging under `logs/large_language_monkeys_<ID>/q1_simple_relation_<ID>.log`.

- You can check the reproducible experimentation process under `workspace/large_language_monkeys_<ID>`.