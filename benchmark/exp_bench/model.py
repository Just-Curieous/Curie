from openai import AzureOpenAI
import os
import json
from typing import Dict
from helper.utils import get_first_n_tokens

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
    
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version=os.getenv('AZURE_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_API_BASE'),
        organization=os.getenv('ORGANIZATION')
    )
    if use_json:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content
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
    
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version=os.getenv('AZURE_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_API_BASE'),
        organization=os.getenv('ORGANIZATION')
    )
    if use_json:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = client.chat.completions.create(
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content
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
        