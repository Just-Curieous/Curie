from openai import AzureOpenAI, OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
import os
import json
import time
from typing import Dict
from helper.utils import get_first_n_tokens


def _make_client():
    """Judge client.

    Harbor parity: an OpenAI-compatible endpoint configured with OPENAI_API_KEY and
    OPENAI_BASE_URL (or OPENAI_API_BASE), as in the Harbor adapter's judge. The model is
    still taken from MODEL. Upstream's AzureOpenAI client is used when only the Azure
    variables (AZURE_API_KEY, AZURE_API_BASE, ...) are set.
    """
    if os.getenv('OPENAI_API_KEY'):
        base_url = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE') or None
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=base_url)
    return AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version=os.getenv('AZURE_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_API_BASE'),
        organization=os.getenv('ORGANIZATION')
    )


def _create_with_retries(client, **kwargs):
    """Transport-level retries only (connection errors, timeouts, 429, 5xx), with the same
    policy as the Harbor adapter's judge; they do not change what is judged."""
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            wait = min(60, 5 * 2**attempt)
            print(f"judge API transient error ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < 5:
                wait = min(60, 5 * 2**attempt)
                print(f"judge API {e.status_code}; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Judge API unavailable after retries")

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
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content or ""  # Harbor parity: None -> unparsable -> default output
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
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
        else:
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None,
                response_format={"type": "json_object"}  
            ) 
    else:
        if "o3" in os.getenv('MODEL') or "o1" in os.getenv('MODEL'):
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                stop=None
            )
        else:
            response = _create_with_retries(client,
                model=os.getenv('MODEL'),
                messages=messages,
                temperature=0.7,
                stop=None
            )

    response_content = response.choices[0].message.content or ""  # Harbor parity: None -> unparsable -> default output
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
        