# LLM Reasoning
## Common setup
Put the API key into `env.sh` under each working directory. 
```
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY=X
export OPENAI_ORGANIZATION=Y
export OPENAI_API_BASE=Z
```

We put all starter file and our code base under `$WORKHOME` 
```
export WORKHOME=~/ # change this to your home repo
```

## Large language monkeys 
```
cd $WORKHOME 
git clone https://github.com/AmberLJC/large_language_monkeys.git
```

Manual testing command after setting up the env:
```
python llmonk/generate/gsm8k.py
python llmonk/evaluate/math_datasets.py  
```

Test on Openhand:
```
source myenv/bin/activate # depends on your setup env
export LOG_ALL_EVENTS=true
cd ~/OpenHands 
poetry run python -m openhands.core.main -f <(cat $WORKHOME/langgraph-exp-agent/benchmark/llm_reasoning/common.txt $WORKHOME/langgraph-exp-agent/benchmark/llm_reasoning/q1_simple_raltion.txt) 2>&1 | tee -a $WORKHOME/langgraph-exp-agent/logs/openhands/llm_reasoning/q1_logging.txt
```


### Tree of thoughts; Scaling test time compute
```
cd $WORKHOME 
git clone https://github.com/AmberLJC/tree-of-thought-llm.git
```

Manual testing command after setting up the env:
```
python run.py --naive_run --task game24 --prompt_sample standard --backend gpt-4
python run.py --n_generate_sample 4 --n_evaluate_sample 1 --n_select_sample 2 --task game24 --backend gpt-4 
```
