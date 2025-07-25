import curie
# Set up your API keys, refer to curie/setup/env.sh.example
key_dict = {
    "MODEL": "azure/gpt-4o",
    "AZURE_API_VERSION": "2025-01-01-preview",
    "AZURE_API_KEY": "bdef9bf150194da5b8ad860046aff989",
    "ORGANIZATION": '327403',
    "AZURE_API_BASE": 'https://api.umgpt.umich.edu/azure-openai-api'
}

result = curie.experiment(api_keys=key_dict, 
                          question="""You are a highly capable researcher. Your job is to solve a given scientific experiment task based on a real paper, which will require you to formulate hypothesis, design an experiment, write and execute experiment code, analyzing results, and produce a conclusion. You are not allowed to read the paper itself (e.g., the PDF of the research paper). You are not allowed to perform any Git operations, such as checking out commits, switching branches, or accessing other versions of the repository. Work only with the current files and contents as given. Do not use the README to obtain results that the question expects you to derive through experimentation. Only reference the README for information not required to be experimentally obtained, such as baseline results or general setup details. Don't just write code, but also execute code, analyze and produce conclusions. Operate strictly within the provided code repo. Save any written code as a file in the repo. 

The task will be provided as input to you in the form of: a question, description of the method, and optionally specific instructions (labelled as "agent_instructions").

Output your response in the following format in valid JSON:
{
  "design": "string or [list, of, strings]",
  "conclusion": "..."
}

Explanation of output keys:
- "design": Describe your experiment design. This could include experiment variables (i.e., independent, dependent, and control variables), or a general outline of the experimental method. Try your best to use the former. Also, design can be specified as a single string or a list of design steps.
- "conclusion": State your final conclusion based on the experiment you conducted, grounded in the results from your code execution. Provide a general relationship or concrete metrics that answers the research question (e.g., numerical improvement, performance gap, statistical significance, etc.).

Input: 

Question: Does the model accurately capture the dynamical content of MD trajectories, specifically in terms of torsion angle relaxation times, autocorrelation behavior at sub-picosecond timescales, and state transition fluxes between metastable states?

Method: #### Problem Setup

- **Objective**: Evaluate the dynamic fidelity of MDGEN-sampled trajectories compared to ground-truth MD, focusing on torsional relaxation timescales, autocorrelation behaviors, and state-level dynamics via MSM fluxes.
- **System**: Tetrapeptide trajectories from all-atom explicit-solvent MD.

#### Independent Variables

- **Trajectory Type**:
  - Ground-truth MD trajectory (100 ns)
  - Model-generated trajectory (100 ns from 10 × 10 ns segments)
- **Torsion Angle Category**:
  - Sidechain torsions
  - Backbone torsions
- **Lag Time Range**:
  - 100 fs to 100 ps

#### Dependent Variables (Evaluation Metrics)

- **Relaxation Times**: Extracted from torsional autocorrelation functions.
- **Autocorrelation Profile**: Including negative derivatives with respect to log-timescale to capture fast dynamic features.
- **MSM Flux Matrix Correlation**: Spearman correlation between flux matrices from MSMs built on MD vs. generated data.

#### Experiment Components

- **Trajectory Generation**:
  - Generate both MD and MDGEN-sampled 100 ns trajectories for a test set of tetrapeptides.
- **Autocorrelation Analysis**:
  1. For each torsion angle (ψ, φ, χ), compute:
      ⟨cos(θ(t) − θ(t + Δt))⟩
      where Δt spans 100 fs to 100 ps.
  2. Estimate **relaxation times** as the lag time where the autocorrelation drops below 1/e of its initial value.
  3. Separate results for:
     - Sidechain torsions
     - Backbone torsions
  4. Compute the **negative derivative** of the autocorrelation with respect to log-timescale to highlight fine dynamic relaxations and sub-picosecond oscillations.
- **Markov State Model (MSM) Flux Comparison**:
  1. Build MSMs by clustering torsion features (via TICA + k-means + PCCA+) from:
     - Ground-truth MD trajectories
     - Model-generated trajectories
  2. Construct **flux matrices** (transition fluxes between metastable states).
  3. Calculate **Spearman correlation** between MD and MDGEN flux matrices.

Agent Instructions: Your task is to implement a system for generating and analyzing molecular dynamics trajectories of peptides. The system consists of two main components:

1. A trajectory generation script that:
   - Takes a pre-trained simulation model checkpoint as input
   - Loads peptide sequence data from a CSV file
   - Generates multiple trajectory rollouts for each peptide sequence
   - Saves the generated trajectories in PDB and XTC formats

2. A trajectory analysis script that:
   - Compares ground-truth MD trajectories with model-generated trajectories
   - Calculates metrics including:
     - Jensen-Shannon divergence between torsion angle distributions
     - Torsion angle autocorrelation functions (separated for backbone and sidechain)
     - Time-lagged independent component analysis (TICA)
     - Markov state models (MSMs) and their transition matrices
   - Optionally generates plots visualizing the comparisons
   - Saves all analysis results to a pickle file

The system should use libraries like MDTraj for trajectory manipulation, PyEMMA for Markov modeling, and standard scientific Python libraries (NumPy, Matplotlib) for calculations and visualization. The analysis should focus on both structural features (torsion angles) and dynamic properties (autocorrelation, state transitions) to evaluate how well the generated trajectories match the ground-truth MD simulations.

The code repo is available in /workspace. LLM related credentials (if needed) are available in /workspace/setup_apis_exp/

Please save your response JSON to: /workspace/93022_task_index_1_iter_1_duration_0.1_eval_gen.json

In addition, you must create a single shell script named `/workspace/reproduce_exp_bench.sh`. This script must reproduce the entire experiment from start to finish, including:
- Any necessary environment setup (e.g., installing dependencies)
- Running the experiment (e.g., other scripts)
- Producing the output or result used in your conclusion

We will use this script to verify your experiment's reproducibility. Make sure it can be run from the root of the repo and reproduces your result end-to-end.""",
                          task_config = {
                            "job_name": "default_research",
                            "docker_image": "amberljc/curie:latest",
                            "dockerfile_name": "ExpDockerfile_pip", 
                            "benchmark_specific_context": "none",
                            "is_user_interrupt_allowed": False,
                            "timeout": 600,
                            "max_coding_iterations": 25,
                            "max_global_steps": 20,
                            #/Users/crakeee/goodgood/iac-project/Curie/
                            "supervisor_system_prompt_filename": "prompts/simple/simple-supervisor.txt",
                            "control_worker_system_prompt_filename": "prompts/simple/simple-control-worker.txt",
                            "patcher_system_prompt_filename": "prompts/simple/simple-patcher.txt",
                            "llm_verifier_system_prompt_filename": "prompts/simple/simple-llm-verifier.txt",
                            "coding_prompt_filename": "prompts/simple/simple-coding.txt",
                            "worker_system_prompt_filename": "prompts/simple/simple-worker.txt",
                            "workspace_name": "workspace/mdgen_eval_task_index_1_iter_1_duration_0.1", # to be filled up by the user
                            "dataset_dir": "", # to be filled up by the user
                            "env_requirements": "", # to be filled up by the user
                            "code_instructions": "", # to be filled up by the user
                            }
                        )