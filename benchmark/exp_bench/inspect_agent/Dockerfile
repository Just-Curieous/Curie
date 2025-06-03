FROM pb-env
# from pb-env, we inherit the env vars:
# constants: WORKSPACE_BASE, SUBMISSION_DIR, LOGS_DIR, CODE_DIR, AGENT_DIR
# overridable: CONDA_ENV_NAME, PYTHON_VERSION, REQUIREMENTS

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${REQUIREMENTS}

WORKDIR ${AGENT_DIR}

# Install all pip packages including the new ones
RUN conda run -n ${CONDA_ENV_NAME} pip install -r requirements.txt && \
    conda run -n ${CONDA_ENV_NAME} pip install inspect-tool-support && \
    conda run -n ${CONDA_ENV_NAME} pip install inspect_ai==0.3.78 && \
    conda run -n ${CONDA_ENV_NAME} pip install openai && \
    conda run -n ${CONDA_ENV_NAME} pip install tiktoken && \
    conda run -n ${CONDA_ENV_NAME} inspect-tool-support post-install && \
    conda clean -afy

RUN mkdir -p /opt/

# Install chz from GitHub
RUN conda run -n ${CONDA_ENV_NAME} pip install "git+https://github.com/openai/chz.git@97cc0dfb5934a4b99c3a96bdcadcfdbe14812fe8#egg=chz"

# copy remaining agent files into ${AGENT_DIR}
COPY . ${AGENT_DIR}