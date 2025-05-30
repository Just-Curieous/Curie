# Faiss
Download Faiss under `starter_file`
```
cd $HOME/langgraph-exp-agent/starter_file
git clone https://github.com/patrickkon/faiss
```

## Commands for Openhands
```
source myenv/bin/activate
export LOG_ALL_EVENTS=true
cd ~/OpenHands 
poetry run python -m openhands.core.main -f <(cat ~/langgraph-exp-agent/benchmark/common.txt /home/ubuntu/langgraph-exp-agent/benchmark/vector_index/q2_hnsw_num_neigh.txt) 2>&1 | tee -a ~/langgraph-exp-agent/logs/openhands/vector_index/q2_logging.txt
```