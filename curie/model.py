from langchain_community.chat_models import ChatLiteLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import time
import os
from openai import BadRequestError
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# import anthropic

import utils

from logger import init_logger
def setup_model_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def report_cost_stats():
    """Print accumulated token usage and costs."""
    stats = TokenCounter.get_accumulated_stats()
    curie_logger.info(f"💸 Accumulated cost for input tokens: ${stats['costs']['input']:.4f}")
    curie_logger.info(f"💸 Accumulated cost for output tokens: ${stats['costs']['output']:.4f}")
    curie_logger.info(f"💸 Accumulated cost for tool usage: ${stats['costs']['tool_cost']:.4f}")

class TokenCounter:
    # Pricing per 1k tokens (as of March 2024)
    PRICE_PER_1K_TOKENS = utils.get_all_price_per_1k_tokens()

    # Class-level variables to track accumulated usage across all instances
    _accumulated_tokens = {"input": 0, "output": 0}
    # Can be more fine-grained once we introduce more tools
    _accumulated_cost = {"input": 0.0, "output": 0.0, "tool_cost": 0.0}

    def __init__(self):
        # Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
        self.model_name = utils.get_model_name()
        
        # try:
        #     self.encoding = tiktoken.encoding_for_model(self.model_name)
        # except KeyError:
        #     # Fall back to cl100k_base for models not in tiktoken
        #     self.encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def get_accumulated_stats(cls) -> Dict[str, Dict[str, float]]:
        """Get accumulated token usage and costs across all instances."""
        return {
            "tokens": dict(cls._accumulated_tokens),
            "costs": dict(cls._accumulated_cost),
            "total_cost": sum(cls._accumulated_cost.values())
        }
    
    def count_output_tokens(self, string: str) -> int:
        """
        MetaGPT anthropic client token counter does not work for anthropic>=0.39.0: https://github.com/geekan/MetaGPT/blob/main/metagpt/utils/token_counter.py#L479C1-L480C1
        Use simple tokenizer instead, since that is what langchain_aws is doing: 
            - https://github.com/langchain-ai/langchain-aws/commit/6355b0ff44c92b594ab8c3a5c50ac726904d716d
            - https://github.com/langchain-ai/langchain-aws/issues/314
            - https://python.langchain.com/api_reference/_modules/langchain_core/language_models/base.html#BaseLanguageModel

        Returns the number of tokens in a text string.

        Args:
            string (str): The text string.

        Returns:
            int: The number of tokens in the text string.
        """
        # if "claude" in self.model_name:
        #     vo = anthropic.Client()
        #     num_tokens = vo.count_tokens(string)
        #     return num_tokens
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # tiktoken only supports oai models
            # curie_logger.debug(f"Warning: model {self.model_name} not found in tiktoken. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        try:
            token_count = len(encoding.encode(string))
            return token_count
        except Exception as e:
            curie_logger.error(f"Error in token counting: {e}")
            return 0

    def count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a single message."""
        num_tokens = self.count_output_tokens(message.content)
        # Add tokens for message format (role, etc.)
        num_tokens += 4  # Format tokens
        return num_tokens

    def count_messages_tokens(self, messages: List[BaseMessage]) -> Dict[str, int]:
        """Count tokens in a list of messages."""
        input_tokens = sum(self.count_message_tokens(msg) for msg in messages)
        return {"input_tokens": input_tokens}

    def estimate_cost(self, token_counts: Dict[str, int]) -> Dict[str, float]:
        """Estimate cost based on token counts, returning costs for both input and output."""
        costs = {"input": 0.0, "output": 0.0}
        
        # Calculate input token cost
        if "input_tokens" in token_counts:
            costs["input"] = (token_counts["input_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS.get(
                self.model_name, {"input": 0.01}
            )["input"]
        
        # Calculate output token cost
        if "output_tokens" in token_counts:
            costs["output"] = (token_counts["output_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS.get(
                self.model_name, {"output": 0.02}
            )["output"]
        
        return costs

    def update_usage(self, token_counts: Dict[str, int]):
        """Update accumulated usage statistics."""
        # Update accumulated tokens
        self._accumulated_tokens["input"] += token_counts.get("input_tokens", 0)
        self._accumulated_tokens["output"] += token_counts.get("output_tokens", 0)
        
        # Calculate and update accumulated costs
        costs = self.estimate_cost(token_counts)
        self._accumulated_cost["input"] += costs["input"]
        self._accumulated_cost["output"] += costs["output"]

def create_model():
    return ChatLiteLLM(model=os.environ.get("MODEL")) 

def create_completion(messages: List[BaseMessage], tools: List = None) -> Any:
    """Create a completion using LiteLLM"""
    try:
        chat = ChatLiteLLM(model=os.environ.get("MODEL"))
        if tools:
            if "anthropic" in os.environ.get("MODEL"):
                chat = chat.bind_tools(tools)
            else:
                chat = chat.bind_tools(tools, parallel_tool_calls=False) # disabling for gpt models
        messages = format_messages(messages)
        return chat.invoke(messages)
    except Exception as e:
        curie_logger.error(f"Error in LLM API create_completion: {e}")
        raise e

def format_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    for msg in messages:
        if msg.content == "":
            msg.content = "Empty message."
    return messages

def update_tool_costs(costs: float):
    """Update accumulated tool usage statistics."""
    TokenCounter._accumulated_cost["tool_cost"] += costs

def get_accumulated_cost() -> float:
    """Get accumulated tool usage cost."""
    return TokenCounter._accumulated_cost 

def query_model_safe(
    messages: List[BaseMessage],
    tools: List = None,
    max_retries: int = 3,
    delay: int = 2
) -> Any:
    """Execute model query with token counting and cost estimation."""
    token_counter = TokenCounter()
    context_length = utils.get_model_context_length()
    max_tokens = context_length - 10000  # Reserve tokens for response
    
    attempt = 0
    while attempt < max_retries:
        try:
            # Count input tokens before processing
            token_counts = token_counter.count_messages_tokens(messages)
            
            # Case 1: Prune messages if total tokens exceed context length
            if token_counts["input_tokens"] > max_tokens:
                curie_logger.info(f"Total tokens ({token_counts['input_tokens']}) exceed limit ({max_tokens}). Pruning messages.")
                start = len(messages) // 3
                end = -len(messages) // 3
                messages_to_prune = messages[start:end]
                j = len(messages_to_prune) - 1
                if messages[end].type == "tool": # edge case where the first message following the last message in messages_to_prune is a tool message. So that means we DO NOT want to prune the last message in messages_to_prune. 
                    messages_to_prune = messages_to_prune[:-1]
                    end -= 1
                    j -= 1
                while j >= 0:
                    if messages_to_prune[j].type == "tool":
                        if j > 0:
                            messages_to_prune.pop()
                            messages_to_prune.pop()
                        j -= 2
                    else:
                        messages_to_prune.pop()
                        j -= 1
                messages = messages[:start] + messages_to_prune + messages[end:]
                token_counts = token_counter.count_messages_tokens(messages)
                curie_logger.info(f"After pruning - Tokens: {token_counts['input_tokens']}")

            # Case 2: Handle large last message
            last_message_tokens = token_counter.count_message_tokens(messages[-1])
            if last_message_tokens > max_tokens // 2:
                curie_logger.info(f"Last message too large ({last_message_tokens} tokens). Splitting and summarizing.")
                chunks = text_splitter_by_tokens(messages[-1].content, max_tokens // 2, token_counter)
                
                if len(chunks) > 1:
                    summarized_text = ""
                    for i, chunk in enumerate(chunks):
                        curie_logger.info(f"Processing chunk {i + 1} of {len(chunks)}")
                        summary_messages = [
                            HumanMessage(
                                content="Summarize the following text. Be concise, but maintain structure. Don't output anything other than the summarized text.\n" + chunk
                            )
                        ]
                        response = create_completion(summary_messages)
                        summarized_text += response.content + "\n"
                        curie_logger.info(f"Chunk {i + 1} summary: {response.content}")
                    
                    messages[-1].content = summarized_text.strip()

                token_counts = token_counter.count_messages_tokens(messages)
                curie_logger.info(f"After summarizing - Tokens: {token_counts['input_tokens']}")

            # Execute final completion
            response = create_completion(messages, tools=tools)
            curie_logger.debug(f"Response: {response}")

            # use tiktoken to count output tokens
            token_counts["output_tokens"] = token_counter.count_output_tokens(response.content)
            token_counter.update_usage(token_counts)            
            # Get current costs
            costs = token_counter.estimate_cost(token_counts)
            accumulated_stats = TokenCounter.get_accumulated_stats()
            # FIXME: this does not count external tool API cost 
            curie_logger.info(f"💰 Round Cost: ${sum(costs.values()):.4f} / Total Cost: ${accumulated_stats['total_cost']:.4f}")

            return response

        except BadRequestError as e:
            curie_logger.error(f"Bad request error: {e}")
            attempt += 1
            if attempt < max_retries:
                curie_logger.error(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
        except Exception as e:
            curie_logger.error(f"Unexpected error: {e}")
            attempt += 1
            if attempt < max_retries:
                curie_logger.error(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
    
    raise RuntimeError(f"Failed after {max_retries} retries.")

# Keep these helper functions unchanged
def text_splitter_by_tokens(text: str, chunk_size: int, token_counter: TokenCounter) -> List[str]:
    """Split text based on token count instead of characters."""
    if isinstance(text, list):
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=lambda x: token_counter.count_output_tokens(x),
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)