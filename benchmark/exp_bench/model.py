from openai import AzureOpenAI, OpenAI
import os
import json
from typing import Dict
from helper.utils import get_first_n_tokens

# Harbor parity: OpenAI-compatible judge endpoint (OPENAI_API_KEY, OPENAI_BASE_URL) when set.
# Claude models via Anthropic's endpoint need a non-empty user turn, a strict JSON schema
# instead of json_object, and no temperature; the prompts are unchanged.
CLAUDE_TYPES = {"design_score": "integer", "setup_score": "integer", "paper_access": "boolean",
                "git_operations": "boolean", "faked_or_nonexperimental_data": "boolean"}

def _make_client():
    if os.getenv('OPENAI_API_KEY'):
        return OpenAI(base_url=os.getenv('OPENAI_BASE_URL') or None, max_retries=6)
    return AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version=os.getenv('AZURE_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_API_BASE'),
        organization=os.getenv('ORGANIZATION')
    )

def _create(client, default_json_output, **kwargs):
    if kwargs["model"].startswith("claude-"):
        kwargs.pop("temperature", None)
        kwargs["messages"] = [m if m["content"] else {**m, "content": "Respond with the JSON object."} for m in kwargs["messages"]]
        if "response_format" in kwargs and default_json_output:
            props = {k: {"type": CLAUDE_TYPES.get(k, "string")} for k in default_json_output}
            if "conclusion_score" in props:
                props["conclusion_score"]["enum"] = ["correct", "incorrect"]
            schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "judge_output", "strict": True, "schema": schema}}
    return client.chat.completions.create(**kwargs)

def llm_query(paper: str, system_prompt: str, 
              default_json_output: Dict = None,
              use_json: bool = True ) -> Dict:
    """Compile explanations for analysis in JSON format.""" 
    # Prepare message for GPT analysis
    paper = get_first_n_tokens(paper)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": paper}
    ]
    
    client = _make_client()
    if use_json:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content or ""
    print("LLM query response: ", response_content)
    # Parse the response as JSON
    try:
        if not use_json:
            return response_content
        else:
            return json.loads(response_content)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        return default_json_output

def llm_query_plain(user_prompt: str, system_prompt: str, 
              default_json_output: Dict = None,
              use_json: bool = True ) -> Dict:
    """Compile explanations for analysis in JSON format.""" 

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    client = _make_client()
    if use_json:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = _create(client, default_json_output,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content or ""
    print("LLM query response: ", response_content)
    # Parse the response as JSON
    try:
        if not use_json:
            return response_content
        else:
            return json.loads(response_content)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        return default_json_output
        