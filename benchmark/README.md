# Benchmark



## Docker Setup for Benchmarking Yourself

```bash
# bin/bash
docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v /Users/jingjia/Curie/curie:/curie:ro \
        -v /Users/jingjia/Curie/benchmark:/benchmark:ro \
        -v /Users/jingjia/Curie/logs:/logs \
        -v /Users/jingjia/Curie/starter_file:/starter_file:ro -v \
        /Users/jingjia/Curie/workspace:/workspace \
        --network=host -d --name exp-agent-container-test exp-agent-image

# set up the environment inside docker container
docker exec -it exp-agent-container-test -c "source /curie/setup/env.sh"
```

