# https://python.langchain.com/docs/how_to/custom_tools/
from langchain_core.tools import tool
from typing import Annotated, List
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import InjectedStore
import uuid
import shutil
from modified_deps.langchain_bash.tool import ShellTool
from typing import Optional, Type, Dict, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from model import update_tool_costs
from utils import load_system_prompt
from model import create_model

import formatter
import settings
import utils

import json
import re
import os

from logger import init_logger
def setup_tool_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

@tool
def test_search_tool(a: Annotated[str, "search string"]) -> str:
    """Searches for useful information"""
    return "University of Michigan!"

# @tool
# def browse_web(query: Annotated[str, "search string"]) -> str:
#     # Other options: https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/
#     return True

shell_tool = ShellTool(timeout=7200)

class CodeAgentInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This was the group that was passed to you as input, it is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    workspace_dir: str = Field(
        ...,
        description="Extract this from the plan JSON's 'workspace_dir' key."
    )
    prompt: str = Field(
        ...,
        description="Clear guidelines on generating workflow/programs."
    )
    dataset_dir: str = Field(
        ...,
        description="Extract this from the plan JSON's 'dataset_dir' key."
    )

    @model_validator(mode="after")
    def partition_name_check(self) -> Self:
        # print("Entering custom model validator: partition_name_check")
        if not utils.extract_partition_name(self.partition_name):
            raise ValueError("partition_name is not specified correctly.")
        return self

    @model_validator(mode="after")
    def plan_id_check(self) -> Self:
        # print("Entering custom model validator: plan_id_check")
        if not utils.extract_plan_id(self.plan_id):
            raise ValueError("plan_id is not specified correctly.")
        return self

    @model_validator(mode="after")
    def workspace_dir_check(self) -> Self:
        # print("Entering custom model validator: workspace_dir_check")
        if not utils.extract_workspace_dir(self.workspace_dir):
            raise ValueError("workspace_dir is not specified correctly.")
        return self 
    
def _collect_openhands_cost():
    total_cost = 0
    # read all openhands log json files under ../logs
    for filename in os.listdir("../logs/openhands"):
        if filename.endswith(".json"):
            remove_flag = False
            with open(f"../logs/openhands/{filename}", "r") as f:
                data = json.load(f)
                if "cost" in data:
                    remove_flag = True
                    total_cost += data["cost"]
            if remove_flag:
                os.remove(f"../logs/openhands/{filename}")
    curie_logger.info(f"$$$$ Total cost of OpenHands: {total_cost} $$$$") 
    update_tool_costs(total_cost)


# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class CodeAgentTool(BaseTool):
    name: str = "codeagent_openhands"
    description: str = "Coding agent that can generate/modify workflow scripts for a given experimentation plan."
    args_schema: Type[BaseModel] = CodeAgentInput
    config: Optional[dict] = None

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = config_dict
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        group: str, 
        partition_name: str, 
        workspace_dir: str, 
        prompt: str,
        dataset_dir: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str: 

        try:  
            utils.setup_openhands_credential() 
            prompt_file_key = "coding_prompt_filename"
            default_prompt_file =  "prompts/exp-coding.txt"
            coding_agent_prompt = self.config.get(prompt_file_key, default_prompt_file)

            # Use a prompt specific to experimental groups, if the user config specifies it:
            exp_prompt_file_key = "exp_group_coding_prompt_filename"
            if exp_prompt_file_key in self.config and group == "experimental_group":
                exp_group_coding_prompt = self.config[exp_prompt_file_key]
                coding_agent_prompt = exp_group_coding_prompt

            system_prompt = load_system_prompt(
                coding_agent_prompt,
                workspace_dir=workspace_dir,
                plan_id=plan_id,
                group=group,
                partition_name=partition_name,
            )
            coding_max_iterations = self.config.get("coding_max_iterations", 30)

            exp_log_dir_parts = self.config["log_filename"].split("/")[:-1]
            exp_log_dir = "/".join(exp_log_dir_parts)
            if dataset_dir:
                prompt += f"\n\nDataset directory: {dataset_dir} (don't create synthetic data, use the real dataset)."
            prompt = f'''{system_prompt}\n{prompt}'''
            curie_logger.info(f"👋👋 Trigger Coding Agent.")
            curie_logger.info(f"🕒 This may take awhile... See log file for details: {exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt")

            # write to a file
            prompt_file = f"../logs/tmp_prompt.txt"
            with open(prompt_file, "w") as file:
                file.write(prompt)

            openhands_dir = self.config["base_dir"] + "/workspace"

            sudo_available = shutil.which("sudo") is not None
            # print("Sudo is available:", sudo_available)
            chmod_cmd = f"{'sudo ' if sudo_available else ''}chmod 777 -R {workspace_dir}"

            # FIXME: remove organization for public use. workspace_base is still hardcoded to home/ubuntu
            output = shell_tool.run({
                "commands": [
                    f"export LOG_ALL_EVENTS=true; "
                    f"{chmod_cmd}; "
                    f"export WORKSPACE_BASE={openhands_dir}; "
                    f"export SANDBOX_TIMEOUT=600; " # FIXME: hardcoded timeout
                    f"/root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/bin/python "
                    f"-m openhands.core.main "
                    f"-f {prompt_file} "
                    f"--config-file ../workspace/config.toml "
                    f"--max-iterations {coding_max_iterations} "
                    f"2>&1 | tee -a /{exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt; "
                ]
            })
            # copy the starter file outside the container to the new directory inside the container
            # FIXME: this does not support running outside the container.
            openhands_log = self.extract_codeagent_output_snippet(
                f"/{exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt"
            )
            curie_logger.info(f"💻 Openhands results: {openhands_log}")
        except BaseException as e:
            curie_logger.error(f"Error for openhands agent: {repr(e)}")
            return f"Failed to generate code for prompt: {prompt}\nError: {repr(e)}"
    
        _collect_openhands_cost()

        return f"""
                The Code Agent has completed. Here's a snippet from the last 10% of the logs —
                use it with the workflow script and results file to evaluate success.
                Re-run the Code Agent with feedback if necessary.
                {openhands_log}
                """.strip()
    
    def extract_codeagent_output_snippet(self, filename: str) -> str:
        """
            Extracts bottom 10% of text within the log filename. 
        """

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            bottom_10_percent = lines[-max(1, len(lines) // 10):]  # Extract bottom 10% of the file
            return "".join(bottom_10_percent)

class PatcherAgentInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This was the group that was passed to you as input, it is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    workspace_dir: str = Field(
        ...,
        description="Extract this from the plan JSON's 'workspace' key."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the controlled experiment workflow."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the controlled experiment workflow."
    )
    prompt: str = Field(
        ...,
        description="Clear guidelines on generating workflow/programs."
    )

    @model_validator(mode="after")
    def partition_name_check(self) -> Self:
        curie_logger.info("Entering custom model validator: partition_name_check")
        if not utils.extract_partition_name(self.partition_name):
            raise ValueError("partition_name is not specified correctly.")
        return self

    @model_validator(mode="after")
    def plan_id_check(self) -> Self:
        curie_logger.info("Entering custom model validator: plan_id_check")
        if not utils.extract_plan_id(self.plan_id):
            raise ValueError("plan_id is not specified correctly.")
        return self

    @model_validator(mode="after")
    def workspace_dir_check(self) -> Self:
        # curie_logger.info("Entering custom model validator: workspace_dir_check")
        if not utils.extract_workspace_dir(self.workspace_dir):
            raise ValueError("workspace_dir is not specified correctly.")
        return self

    @model_validator(mode="after")
    def workflow_file_check(self) -> Self:
        # curie_logger.info("Entering custom model validator: workflow_file_check")
        if f"{self.workspace_dir}/control_experiment_{self.plan_id}_{self.group}_{self.partition_name}.sh" != self.control_experiment_filename:
            raise ValueError("control_experiment_filename is not specified correctly.")
        return self

    @model_validator(mode="after")
    def workflow_results_file_check(self) -> Self:
        # curie_logger.info("Entering custom model validator: workflow_results_file_check")
        if f"{self.workspace_dir}/results_{self.plan_id}_{self.group}_{self.partition_name}.txt" != self.control_experiment_results_filename:
            raise ValueError("control_experiment_results_filename is not specified correctly.")
        return self

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class PatcherAgentTool(BaseTool):
    name: str = "patchagent_openhands"
    description: str = "Coding agent that can patch incorrect workflow scripts for a given experimentation plan."
    args_schema: Type[BaseModel] = PatcherAgentInput
    config: Optional[dict] = None

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = config_dict
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        group: str, 
        partition_name: str, 
        workspace_dir: str, 
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        prompt: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
    # given the code workspace 
    # wait; return log and result script 
        try:   
            coding_max_iterations = self.config.get("coding_max_iterations", 30)
            prompt_file_key = "coding_patch_prompt_filename"
            default_prompt_file =  "prompts/exp-patch-coding.txt"
            coding_agent_prompt = self.config.get(prompt_file_key, default_prompt_file)
            system_prompt = load_system_prompt(
                coding_agent_prompt,
                workspace_dir=workspace_dir,
                plan_id=plan_id,
                group=group,
                partition_name=partition_name,
                control_experiment_filename=control_experiment_filename,
                control_experiment_results_filename=control_experiment_results_filename

            )
            
            exp_log_dir = f"logs/{self.config['workspace_name']}_{self.config['unique_id']}_iter{self.config['iteration']}"
            prompt = f'''{system_prompt}\n{prompt}'''
            curie_logger.info(f"👋👋 Trigger Coding Patch Agent.")
            curie_logger.info(f"🕒 This may take awhile... See log file for details: {exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt")
            # write to a file
            prompt_file = f"../logs/tmp_prompt.txt"
            with open(prompt_file, "w") as file:
                file.write(prompt)

            openhands_dir = os.path.join(self.config["base_dir"], "workspace")
            sudo_available = shutil.which("sudo") is not None
            chmod_cmd = f"{'sudo ' if sudo_available else ''}chmod 777 -R {workspace_dir}"
            
            output = shell_tool.run({
                "commands": [
                    f"export LOG_ALL_EVENTS=true; "
                    f'sed -i "474i \          \'organization\': \'499023\'," /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/azure.py; '
                    f"{chmod_cmd}; "
                    f"export WORKSPACE_BASE={openhands_dir}; "
                    f"export SANDBOX_TIMEOUT=600; " # FIXME: hardcoded timeout
                    f"/root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/bin/python -m openhands.core.main -f {prompt_file} --config-file ../workspace/config.toml --max-iterations {coding_max_iterations} 2>&1 | tee -a /{exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt; " # TODO: create a new file for each openhands log (important to prevnet simultaneous writes in parallel exec situations).
                ]
            }) 
            # read log
            openhands_log = self.extract_codeagent_output_snippet(
                f"/{exp_log_dir}/openhands_{plan_id}_{group}_{partition_name}_logging.txt"
            )
            curie_logger.info(f"💻 Openhands results: {openhands_log}")
            # copy the starter file outside the container to the new directory inside the container
            # FIXME: this does not support running outside the container. 

        except BaseException as e:
            curie_logger.info(f"Error for openhands agent: {repr(e)}")
            return f"Failed to generate code for prompt: {prompt}\nError: {repr(e)}"
        # return (f"Workflow and results have been produced, for plan_id: {plan_id}, group: {group}, partition_name: {partition_name} \n"
        #         f"control_experiment_filename is at: '{workspace_dir}/control_experiment_{plan_id}_{group}_{partition_name}.sh'\n"
        #         f"Control group results are stored in '{workspace_dir}/results_{plan_id}_{group}_{partition_name}.txt'\n"
        #         f"[Minor] Openhands logging can be found in '/logs/openhands_{plan_id}_{group}_{partition_name}_logging.txt'"
        #         )
        _collect_openhands_cost()

        return f"""
            Patch Agent has completed. Here's a snippet of the latest logs—
            use this along with the workflow script and results file to assess success.
            Re-run the Patch Agent with feedback if needed.

            {openhands_log}
            """.strip()


    def extract_codeagent_output_snippet(self, filename: str) -> str:
        """
            Extracts bottom 10% of text within the log filename. 
        """

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            bottom_10_percent = lines[-min(20, len(lines)):]  # Extract bottom 10% of the file
            # bottom_10_percent = lines[-max(1, len(lines) // 10):]  # Extract bottom 10% of the file
            return "".join(bottom_10_percent)

# Note: shell_tool itself can in theory be passed into the agent as a tool already https://python.langchain.com/docs/integrations/tools/bash/ https://www.youtube.com/watch?v=-ybgQK0BE-I
@tool
def execute_shell_command(
    command: Annotated[str, "The shell command to execute"]
    ) -> str:
    """Execute one line shell command and return the output."""
    try:
        # For all $ symbols in the command, that don't have a \ appended right before the $, add a \ right before the $ symbol:
        command = re.sub(r'(?<!\\)\$', r'\\$', command)

        if "ls -lR" in command or "ls -R" in command:
            return "Please don't use 'ls -lR' or 'ls -R' commands. They are not allowed, as they will cause you to exceed context length."

        curie_logger.info(f"🐚 Running command: {command}")
        output = shell_tool.run({"commands": [command]}) # only run one command at a time 
        curie_logger.info(f"🐚 Output: {output}")
        # cut the output to last 1000 characters - 200 tokens
        if len(output) > 2500:
            output = output[:1000] + '...(omitted for brevity)...' + output[-1000:]

    except BaseException as e:
        curie_logger.error(f"Error executing command: {command}")
        curie_logger.error(f"Error: {repr(e)}")
        return f"Failed to execute command: ```bash\n{command}\n```\nError: {repr(e)}"
    return f"Command executed: ```bash\n{command}\n```\nStdout: {output}"

@tool
def write_to_file(
    input_string: Annotated[str, "The string to write to the file"],
    file_path: Annotated[str, "The path of the file where the string will be written"]
) -> str:
    """Write a given string to a file line by line after processing it."""
    try:
        # Split the string into lines
        lines = input_string.splitlines()  # Preserve actual newlines

        with open(file_path, "w") as file:
            for line in lines:
                # Replace literal '\n' with '\\n'
                processed_line = line.replace("\n", "\\n")
                curie_logger.info(f"🔧 Writing line: {processed_line}")
                file.write(processed_line + "\n")

        return f"String written successfully to {file_path}"
    except BaseException as e:
        curie_logger.error(f"Error writing to file: {file_path}")
        curie_logger.error(f"Error: {repr(e)}")
        return f"Failed to write to file: {file_path}\nError: {repr(e)}"

@tool
def read_file_contents(
    filename: Annotated[str, "The absolute path of the file to read"]
    ) -> str:
    """
    Reads and returns the contents of a file.

    Parameters:
        filename (str): The absolute path of the file to read.

    Returns:
        str: The contents of the file as a string.
    """
    try:
        if not os.path.exists(filename):
            target = os.path.basename(filename) 
            # may also under /workspace/ need to specify the workspace name
            root_dir_list = ['/starter_file/', '/workspace/']   
            # Recursively walk through directory
            find_flag = False
            for root_dir in root_dir_list:
                for root, dirs, files in os.walk(root_dir):
                    if target in files:
                        full_path = os.path.join(root, target)
                        print(f"Found {target} at: {full_path}")
                        filename = full_path
                        find_flag = True
                        break
                if find_flag:
                    break

        curie_logger.info(f"🔧 Reading file: {filename}")

        with open(filename, 'r') as file:
            # content = file.read()
            # FIXME: only read first 100 lines
            content = "".join(file.readlines()[:25])

        return content
    except FileNotFoundError:
        return f"Error: The file '{filename}' does not exist."
    except PermissionError:
        return f"Error: Insufficient permissions to read the file '{filename}'."
    except Exception as e:
        return f"Error: An unexpected error occurred while reading the file: {str(e)}"


document_cache = {}
index_cache = {}
class QueryPDFInput(BaseModel):
    question: str = Field(
        ...,
        description="The question to answer about the PDF content"
    )
    pdf_path: str = Field(
        ...,
        description="Path to the PDF file"
    ) 

class QueryPDFTool(BaseTool):
    name: str = "query_pdf"
    description: str = "Read or Answer a question about a PDF file."
    args_schema: Type[BaseModel] = QueryPDFInput
    config: Optional[dict] = None

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = config_dict

    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        question: str, 
        pdf_path: str,
        workspace_dir: str = None, 
        plan_id: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict: 
        if plan_id is None or workspace_dir is None:
            pdf_dir = '/starter_file/' + self.config["workspace_name"]
            pdf_path = os.path.join(pdf_dir, pdf_path)
        else:
            # this assume the pdf is put under the outer workspace dir
            pdf_path = os.path.basename(pdf_path)
            pdf_path = os.path.join(workspace_dir, pdf_path)
        
        if not os.path.exists(pdf_path):
            target = os.path.basenamev(pdf_path) 
            root_dir = '/starter_file/' + self.config["workspace_name"]
  
            # Recursively walk through directory
            for root, dirs, files in os.walk(root_dir):
                if target in files:
                    full_path = os.path.join(root, target)
                    print(f"Found {target} at: {full_path}")
                    pdf_path = full_path
                    break

        curie_logger.info(f"Querying PDF: {pdf_path} with question: {question}")
        try:
            result = query_pdf(question, pdf_path)
        except Exception as e:
            curie_logger.error(f"Error querying PDF: {str(e)}")
            return {"error": f"Error querying PDF: {str(e)}"}
        return result

# @tool
def load_pdf(pdf_path: str) -> dict: 
    if not os.path.exists(pdf_path):
        curie_logger.error(f"PDF file not found at {pdf_path}")
        return {"error": f"PDF file not found at {pdf_path}"}
    
    try:
        curie_logger.info(f"Loading PDF: {pdf_path}")
        # Check if document is already in cache
        if pdf_path in document_cache:
            return {
                "status": "success", 
                "message": f"PDF '{pdf_path}' was already loaded and indexed",
                "pages": len(document_cache[pdf_path])
            }
    
        loader = PyPDFLoader(pdf_path)
        documents = loader.load() 
        document_cache[pdf_path] = documents

        # Create text chunks for better indexing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Create a vector index
        if 'ORGANIZATION' in os.environ:
            endpoint = os.environ['AZURE_API_BASE'] 
            if "AZURE_API_BASE" in os.environ:
                del os.environ["AZURE_API_BASE"] 
            if "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]

            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint= endpoint,
                openai_api_version="2024-06-01",  
                openai_api_key=os.environ['AZURE_API_KEY'],   
                openai_organization=os.environ['ORGANIZATION'],
                model="text-embedding-3-large" ,
            )
            os.environ["AZURE_API_BASE"] = endpoint
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large",)
        vector_index = FAISS.from_documents(chunks, embeddings) 
        index_cache[pdf_path] = vector_index
        curie_logger.info(f"{pdf_path} statistics: {len(documents)} pages, {len(chunks)} chunks")
        return {
            "status": "success",
            "message": f"PDF '{pdf_path}' has been loaded and indexed successfully",
            "pages": len(documents),
            "chunks": len(chunks)
        }
    except Exception as e:
        return {"error": f"Error indexing PDF: {str(e)}"}


def query_pdf(question: str, pdf_path: str) -> dict:
    """
    Answer questions about a specific PDF file that has been loaded.
    
    Args:
        pdf_path: Path to the previously loaded PDF file
        question: The question to answer about the PDF content
        
    Returns:
        Dictionary with the answer and relevant sources
    """
    if pdf_path not in index_cache:
        # Try to load the PDF first
        try: 
            load_result = load_pdf(pdf_path)
        except Exception as e:
            curie_logger.error(f"Error loading PDF: {str(e)}")
            return {"error": f"Error loading PDF: {str(e)}"}
        if "error" in load_result:
            return {"error": f"PDF must be loaded first: {load_result['error']}"}
    
    try:
        curie_logger.info(f"Building index for PDF: {pdf_path}")
        # Get the index
        vector_index = index_cache[pdf_path]
        
        # Create a question-answering chain
        llm = create_model()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_index.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        # Run the query
        result = qa_chain({"query": question})
        
        # Format source documents
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "...",  # First 200 chars
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", pdf_path)
            })
        curie_logger.info(f"PDF query result: {result['result']}")
        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        return {"error": f"Error querying PDF: {str(e)}"}


# Long term memory: https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/
# i.e., in memory store: https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store
# API reference: https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get
class NewExpPlanStoreWriteInput(BaseModel):
    plan: formatter.NewExperimentalPlanResponseFormatter = Field(
        ...,
        description="Experimental plan dict to store in long-term storage. This field is required."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class NewExpPlanStoreWriteTool(BaseTool):
    name: str = "write_new_exp_plan"
    description: str = "Write a new experimental plan to long term storage."
    args_schema: Type[BaseModel] = NewExpPlanStoreWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan: formatter.NewExperimentalPlanResponseFormatter, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        curie_logger.info("Writing new plan 📖 ")

        memory_id = str(uuid.uuid4())
        # Add metadata and reformat plan:
        plan_data = plan.dict()
        plan_data = self.add_plan_metadata(plan_data)

        # Add plan to store:
        plan_data["plan_id"] = memory_id
        self.async_notify_sched_modify_plan(memory_id)
        self.store.put(namespace, memory_id, plan_data)

        return memory_id
    
    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

    def add_plan_metadata(self, plan_data: dict) -> dict:
        """ This partitions and adds required metadata to the plan. 
            Example modified plan: # this is the only plan format ever saved in long term store. 
            {...all fields except for control and experimental group, 
                "question": "What is the best instance type for a CPU workload?", # original user_input
                "workspace_dir": /workspace/{config["workspace_name"]}_{plan_id},
                "control_group": {
                    "partition_1": {
                        "independent_vars": [{
                            "region": "us-east-1",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_control_group.txt",
                        "done": False, 
                        "error_feedback": "", 
                    }
                },
                "experimental_group": {
                    "partition_1": {
                        "independent_vars": [{
                            "region": "us-east-1",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
                        "done": False, 
                        "error_feedback": "", 
                    },
                    "partition_2": {
                        "independent_vars": [{
                            "region": "us-west-2",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
                        "done": False, 
                        "error_feedback": "", 
                    }
                }
            }

            (just for reference) Archived outdated plan format:
            {..., "experimental_group":{"vcpu":[1,2,3,4]}, "experimental_group_partition_1":{"vcpu":[1,2]}, "experimental_group_partition_2":{"vcpu":[3,4]}, "experimental_group_partition_1_done: False, "experimental_group_partition_2_done: False, control_group_done: False}
        """
        curie_logger.info("Add Plan Metadata. ")

        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now
        memory_id = str("question")
        question = self.metadata_store.get(sched_namespace, memory_id)
        question = question.dict()["value"]

        partitioned_plan_data = {"control_group": {}, "experimental_group": {}, "question": question, "workspace_dir": ""} # note: workspace will be assigned in sched later
        # Partition applied to experimental groups only for now:
        for key, value in plan_data.items():
            if key != "experimental_group" and key != "control_group":
                partitioned_plan_data[key] = value                
            else:
                group_type = key
                # print(key)
                num_exp_groups = len(plan_data[group_type]) # using the example, this would be 4
                i = 0
                partition_count = 0
                while i < num_exp_groups:
                    var_group = []
                    for j in range(settings.VARS_PER_PARTITION):  
                        if i + j >= num_exp_groups:
                            break
                        var_group.append(plan_data[group_type][j+i])

                    partitioned_plan_data[group_type][f"partition_{partition_count+1}"] = {
                        "independent_vars": var_group,
                        "control_experiment_filename": "",
                        "control_experiment_results_filename": "",
                        "all_control_experiment_results_filename": "",
                        "done": False,
                        # "error_feedback": ""
                    }
                    
                    i += settings.VARS_PER_PARTITION
                    partition_count += 1

        return partitioned_plan_data

class ExistingExpPlanStoreWriteInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    plan: formatter.ExistingExperimentalPlanResponseFormatter = Field(
        ...,
        description="Experimental plan dict to store in long-term storage. This field is required."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ExistingExpPlanStoreWriteTool(BaseTool):
    name: str = "edit_existing_exp_plan"
    description: str = "Modify existing experimental plan in long term storage. Warning: This enables overwriting any field within the plan, try avoiding this tool if possible (if other suitable tools are available)."
    args_schema: Type[BaseModel] = ExistingExpPlanStoreWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan: formatter.ExistingExperimentalPlanResponseFormatter, 
        plan_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        curie_logger.info("Modifying existing plan...")
        memory_id = plan_id
        plan_data = plan.dict()
        # remove existing plan:
        self.store.delete(namespace, memory_id)

        # Add plan to store:
        self.async_notify_sched_modify_plan(memory_id)
        self.store.put(namespace, memory_id, plan_data)

        return memory_id
    
    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

class RedoExpPartitionInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    group: str = Field(
        None,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        None,
        description="The partition_name belonging to the group that needs re-doing."
    )
    error_feedback: str = Field(
        None,
        description="Feedback to the lab technician on what needs to be fixed to get the partition working."
    )

    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class RedoExpPartitionTool(BaseTool):
    name: str = "redo_exp_partition"
    description: str = "Redo a specific partition of a group in an existing experimental plan. This should be used if you believe the partition's experimental workflow is incorrect, or not working as expected."
    args_schema: Type[BaseModel] = RedoExpPartitionInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        error_feedback: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now
        
        curie_logger.info("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        # plan[group][partition_name]["error_feedback"] = error_feedback
        plan[group][partition_name]["done"] = False
        plan[group][partition_name]["all_control_experiment_results_filename"] = "" # reset this since the current exec verifiers results will not be needed. # NOTE: we may choose to preserve all history in the future

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_redo_partition(plan_id, group, partition_name, error_feedback)

        return "Record partition for redo successful."

    def async_notify_sched_redo_partition(self, plan_id: str, group: str, partition_name: str, error_feedback: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_redo_partition_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "error_feedback": error_feedback
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

class RemoveExpPartitionInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    group: str = Field(
        None,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        None,
        description="The partition_name belonging to the group that needs its independent variables changed."
    )
    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class RemoveExpPartitionTool(BaseTool):
    name: str = "remove_exp_plan_partition"
    description: str = "Remove a partition from some existing experimental plan."
    args_schema: Type[BaseModel] = RemoveExpPartitionInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        curie_logger.info("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        if group not in plan:
            return "The group provided does not exist in the plan."
        if partition_name not in plan[group]:
            return "The partition_name provided does not exist in the group."
        
        del plan[group][partition_name]

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_modify_plan(plan_id)

        return "Edit priority successful."

    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

class ArchiveExpPlanInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )

class ArchiveExpPlanTool(BaseTool):
    name: str = "exp_plan_archive"
    description: str = "Remove an existing experimental plan."
    args_schema: Type[BaseModel] = ArchiveExpPlanInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        curie_logger.info("Modifying existing plan...")

        # Get plan from memory
        try:
            plan = self.store.get(plan_namespace, plan_id).dict()["value"]
        except:
            return "The plan does not exist."

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        self.del_sched_metadata(plan_id)

        return "Plan removal successful."

    def del_sched_metadata(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        # Remove existing plan from supervisor_wrote_list:
        memory_id = str("supervisor_wrote_list")
        supervisor_wrote_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for plan_id2 in supervisor_wrote_list:
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            if plan_id2 != plan_id:
                new_list.append(plan_id2)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

        # Remove existing plan from standby_exp_plan_list:
        memory_id = str("standby_exp_plan_list")
        standby_exp_plan_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for plan_id2 in standby_exp_plan_list:
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            if plan_id2 != plan_id:
                new_list.append(plan_id2)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

        # Remove existing plan from supervisor_redo_partition_list:
        memory_id = str("supervisor_redo_partition_list")
        supervisor_redo_partition_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for redo_details in supervisor_redo_partition_list:
            if redo_details["plan_id"] != plan_id:
                new_list.append(redo_details)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

class EditExpPriorityInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    # group: str = Field(
    #     None,
    #     description="This is either 'control_group' or 'experimental_group'."
    # )
    # partition_name: str = Field(
    #     None,
    #     description="The partition_name belonging to the group that needs its independent variables changed."
    # )
    priority: int = Field(
        None, gt=0,
        description="An integer representing the priority of the experiment. Lower values indicate higher priority."
    )
    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class EditExpPriorityTool(BaseTool):
    name: str = "edit_exp_plan_priority"
    description: str = "Modify the priority of an existing experimental plan."
    args_schema: Type[BaseModel] = EditExpPriorityInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        priority: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        # print(state["prev_agent"])
        
        curie_logger.info("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        plan["priority"] = priority

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_modify_plan(plan_id)

        return "Edit priority successful."

    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        # wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        # wrote_list = wrote_list.dict()["value"]
        # print(wrote_list)

# class EditExpPartitionInput(BaseModel):
#     plan_id: str = Field(
#         None,
#         description="Plan ID of the specific experimental plan from storage."
#     )
#     group: str = Field(
#         None,
#         description="This is either 'control_group' or 'experimental_group'."
#     )
#     partition_name: str = Field(
#         None,
#         description="The partition_name belonging to the group that needs its independent variables changed."
#     )
#     independent_vars: List[Dict[str, Any]] = Field(
#         None,
#         description="New independent variables to replace the current one for this partition."
#     )

#     # state: Annotated[dict, InjectedState]

# # Note: It's important that every field has type hints. BaseTool is a
# # Pydantic class and not having type hints can lead to unexpected behavior.
# class EditExpPartitionsTool(BaseTool):
#     name: str = "edit_exp_partition_independent_vars"
#     description: str = "Modify the independent vars of a specific partition of a group in an existing experimental plan. Use with caution: this will discard all previous results and state related to the partition."
#     args_schema: Type[BaseModel] = EditExpPartitionInput
#     # Holy cow this was frustrating to figure out... 
#     # None of the following work:
#     # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
#     # https://github.com/langchain-ai/langchain/discussions/24906
#     # and so on..
#     store: Optional[InMemoryStore] = None  # Declare store as an optional field
#     metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

#     def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
#         super().__init__()
#         self.store = store
#         self.metadata_store = metadata_store
#         # self.user_id = "admin"
#         # self.application_context = "exp-plans" 
#         # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
#     class Config:
#         arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

#     def _run(
#         self, 
#         # state: Annotated[dict, InjectedState], 
#         plan_id: str, 
#         group: str,
#         partition_name: str,
#         error_feedback: str,
#         run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """
#         Use the tool.
        
#         Note: we are guaranteed that plan will conform to the required format
#         """
#         user_id = "admin"
#         application_context = "exp-plans" 
#         plan_namespace = (user_id, application_context) # just a random namespace name for now

#         # print(state["prev_agent"])
        
#         print("Modifying existing plan...")

#         # Get plan from memory
#         plan = self.store.get(plan_namespace, plan_id).dict()["value"]

#         plan[group][partition_name]["error_feedback"] = error_feedback
#         plan[group][partition_name]["done"] = False

#         # Remove existing plan:
#         self.store.delete(plan_namespace, plan_id)

#         # Add plan to store:
#         self.store.put(plan_namespace, plan_id, plan)

#         self.async_notify_sched_redo_partition(plan_id, group, partition_name, error_feedback)

#         return "Success."

#     def async_notify_sched_modify_plan(self, plan_id: str):
#         user_id = "admin"
#         application_context = "exp-sched" 
#         sched_namespace = (user_id, application_context) # just a random namespace name for now

#         memory_id = str("supervisor_wrote_list")
    
#         wrote_list = self.metadata_store.get(sched_namespace, memory_id)
#         wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
#         # print(wrote_list)
#         wrote_list.append(plan_id)
#         self.metadata_store.put(sched_namespace, memory_id, wrote_list)

#         # wrote_list = self.metadata_store.get(sched_namespace, memory_id)
#         # wrote_list = wrote_list.dict()["value"]
#         # print(wrote_list)

class StoreGetInput(BaseModel):
    plan_id: Optional[str] = Field(
        None,
        description="Plan ID to retrieve the specific experimental plan from storage. Optional. Provide if modifying an existing plan. If not provided, all existing plans will be retrieved."
    )
    # state: Annotated[dict, InjectedState]

class StoreGetTool(BaseTool):
    name: str = "exp_plan_get"
    description: str = "Get all experimental plans, or get a specific experimental plan by plan_id, from long term storage."
    args_schema: Type[BaseModel] = StoreGetInput
    store: Optional[InMemoryStore] = None  # Declare store as an optional field

    def __init__(self, store: InMemoryStore):
        super().__init__()
        self.store = store # Ensure that the same store is passed across all tools and agents
        
    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: Optional[str] = None, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        if not plan_id: # get all plans
            items = self.store.search(namespace)
        else: # specific plan
            # list "memories" within this namespace, filtering on content equivalence
            items = self.store.search(namespace, filter={"plan_id": plan_id})

        # Unpack all items and return them as a list
        return [item.dict()["value"] for item in items] # Cleaner return: only return the "value" key, since the "value" key's value (which presents the plan dict) contains plan-key, we don't need the other attributes        

    # async def _arun(
    #     self,
    #     a: int,
    #     b: int,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     # If the calculation is cheap, you can just delegate to the sync implementation
    #     # as shown below.
    #     # If the sync calculation is expensive, you should delete the entire _arun method.
    #     # LangChain will automatically provide a better implementation that will
    #     # kick off the task in a thread to make sure it doesn't block other async code.
    #     return self._run(a, b, run_manager=run_manager.get_sync())

# # @tool
# def store_write_tool(a: Annotated[str, "search string"], store) -> str:
#     """Stores useful information"""
#     # InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
#     # store = InMemoryStore()
#     user_id = "admin"
#     application_context = "exp-plans" 
#     namespace = (user_id, application_context) # just a random namespace name for now
#     memory_id = str(uuid.uuid4())
    
#     memory = {"rules": ["User likes short, direct language", "User only speaks English & python"], "my-key": "my-value"}
#     store.put(namespace, memory_id, memory)
#     # get the "memory" by ID
#     item = store.get(namespace, memory_id)
#     # list "memories" within this namespace, filtering on content equivalence
#     items = store.search(namespace, filter={"my-key": "my-value"})

#     return memory_id

# def store_get_tool(memory_id, store):
#     # store = InMemoryStore()
#     user_id = "admin"
#     application_context = "exp-plans" 
#     namespace = (user_id, application_context) # just a random namespace name for now
#     # memory_id = "9b177ab7-d35e-4cca-b5a0-e5af08bfbeb2"
#     item = store.get(namespace, memory_id)
#     print(item)
#     # list "memories" within this namespace, filtering on content equivalence
#     items = store.search(namespace, filter={"my-key": "my-value"})
#     print(items[-1].dict())

# store = InMemoryStore()
# memory_id = store_write_tool("test", store)
# store_get_tool(memory_id, store)

class ExpPlanCompletedWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This was the group that was passed to you as input, it is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the controlled experiment workflow."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the controlled experiment workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ExpPlanCompletedWriteTool(BaseTool):
    name: str = "exp_plan_partition_done_write"
    description: str = "Write the completed partition of an experimental plan's group to long term storage."
    args_schema: Type[BaseModel] = ExpPlanCompletedWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        print("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        plan[group][partition_name]["done"] = True
        plan[group][partition_name]["control_experiment_filename"] = control_experiment_filename
        plan[group][partition_name]["control_experiment_results_filename"] = control_experiment_results_filename

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        return "Successfully recorded the completion of this partition."

class LLMVerifierWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the control experiment, that was passed to you as input."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the control experiment workflow."
    )
    is_correct: bool = Field(
        ...,
        description="Indicates whether the experimental workflow is correct."
    )
    verifier_log_message: str = Field(
        ...,
        description="Log or error message describing any issues with the experimental workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class LLMVerifierWriteTool(BaseTool):
    name: str = "workflow_verified_record"
    description: str = "Record workflows that you have evaluated to long term storage."
    args_schema: Type[BaseModel] = LLMVerifierWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        is_correct: bool,
        verifier_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("llm_verifier_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "control_experiment_filename": control_experiment_filename,
            "control_experiment_results_filename": control_experiment_results_filename,
            "is_correct": is_correct,
            "verifier_log_message": verifier_log_message
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class PatchVerifierWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the control experiment, that was passed to you as input."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the control experiment workflow."
    )
    is_correct: bool = Field(
        ...,
        description="Indicates whether the experimental workflow is correct."
    )
    patcher_log_message: str = Field(
        ...,
        description="Log or error message describing any issues with the experimental workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class PatchVerifierWriteTool(BaseTool):
    name: str = "workflow_patched_record"
    description: str = "Record workflows that you have evaluated to long term storage."
    args_schema: Type[BaseModel] = PatchVerifierWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        is_correct: bool,
        patcher_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("patch_verifier_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "control_experiment_filename": control_experiment_filename,
            "control_experiment_results_filename": control_experiment_results_filename,
            "is_correct": is_correct,
            "patcher_log_message": patcher_log_message
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class AnalyzerWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    no_change: bool = Field(
        ...,
        description="No change means that after analyzing the partition’s results, you determine that the existing plan and partitions remain valid, requiring no modifications or new plan creation."
    )
    analyzer_log_message: str = Field(
        ...,
        description="Log message detailing the analysis performed and decisions made along with justification."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class AnalyzerWriteTool(BaseTool):
    name: str = "analyzer_record"
    description: str = "Record partitions that you have analyzed to long term storage."
    args_schema: Type[BaseModel] = AnalyzerWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        no_change: bool,
        analyzer_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("analyzer_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "no_change": no_change,
            "analyzer_log_message": analyzer_log_message,
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class ConcluderWriteInput(BaseModel):
    is_conclude: bool = Field(
        ...,
        description="This means you decide to conclude the experiment, i.e. after analyzing all partitions results, you determine that the existing plans and partitions remain valid, requiring no modifications or new plan creation."
    )
    concluder_log_message: str = Field(
        ...,
        description="Log message detailing the analysis performed and decisions made along with justification."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ConcluderWriteTool(BaseTool):
    name: str = "concluder_record"
    description: str = "Record partitions that you have analyzed to long term storage."
    args_schema: Type[BaseModel] = ConcluderWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        is_conclude: bool,
        concluder_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("concluder_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        wrote_list.append({
            "is_conclude": is_conclude,
            "concluder_log_message": concluder_log_message,
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class UserInputRouterWriteInput(BaseModel):
    is_correct: bool = Field(
        ...,
        description="A value of True means the user decides that the architect's plan is good to proceed."
    )
    router_log_message: str = Field(
        ...,
        description="If plan is incorrect, describe in detail what the user wants changed."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class UserInputRouterWriteTool(BaseTool):
    name: str = "user_router_record"
    description: str = "Record your analysis of user input to long term storage."
    args_schema: Type[BaseModel] = UserInputRouterWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        is_correct: bool,
        router_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("user_router_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        wrote_list.append({
            "is_correct": is_correct,
            "router_log_message": router_log_message,
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."