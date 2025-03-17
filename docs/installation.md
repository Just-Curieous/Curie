# Installation

1. Install docker: https://docs.docker.com/engine/install/ubuntu/. 

  - Grant permission to docker via `sudo chmod 666 /var/run/docker.sock`. 

  - If you encounter an error that `/var/run/docker.sock` doesn’t exist, you may find the actual path to `docker.sock` and create a soft link. For example, Docker Desktop stores this file at `~/.docker/desktop/docker.sock`, in which case you may use:

    ```bash
    sudo chmod 666 ~/.docker/desktop/docker.sock
    sudo ln -s ~/.docker/desktop/docker.sock /var/run/docker.sock
    ```

  - Run `docker ps` to check that permission has been granted with the Docker daemon.

2. Clone the repository:

```
git clone https://github.com/Just-Curieous/Curie.git
cd Curie
```

3. Put your LLM API credentials under `curie/setup/env.sh`. Example: 

```
export MODEL="gpt-4o" 
export OPENAI_API_KEY="sk-xxx" 
```

4. Build the container image. This will take a few minutes. Note: you may need to setup a virtual environment before running pip install.

```bash
pip install -e .
docker images -q exp-agent-image | xargs -r docker rmi -f # remove any existing conflict image
cd curie && docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
```
