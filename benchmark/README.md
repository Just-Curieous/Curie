# Benchmark

## Docker Setup for Benchmarking Yourself

Change to the path of Curie project, then

```bash
# bin/bash
docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v $(pwd)/curie:/curie:ro \
        -v $(pwd)/benchmark:/benchmark:ro \
        -v $(pwd)/logs:/logs \
        -v $(pwd)/starter_file:/starter_file:ro \
        -v $(pwd)/workspace:/workspace \
        --network=host -d --name exp-agent-container-test exp-agent-image
```

# set up the environment inside docker container

docker exec -it exp-agent-container-test -c "source /curie/setup/env.sh"

```

