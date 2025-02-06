from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import os
from openai import BadRequestError

GPT_4O_MAX_CHARS = 128000 * 4 - 10000 # an approximation based on gpt-4o context length, which is 128000 tokens, and also because 4 char is around 1 token, we minus 10000 for safety
GPT_4O_DEFINED_CHAR_LIMIT = 100000 * 2 # giving ourselves more leeway than the actual limit, to avoid encountering case 1 as much as possible

def create_gpt_4(tools=None) -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
        model='gpt-4o',
        api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('OPENAI_API_BASE'),
        api_version=os.getenv('API_VERSION'),
        organization=os.getenv('OPENAI_ORGANIZATION'),
    )

    if tools:
        llm = llm.bind_tools(
            tools, 
            parallel_tool_calls=False # Avoid multi_tool_use.parallel in GPT-4o
        )

    return llm

def text_splitter_gpt_4o(text, chunk_size=GPT_4O_MAX_CHARS):
    # Context window related:
    # https://github.com/langchain-ai/langchain/issues/1349#issuecomment-1521567675
    # GPT's context window and max output tokens: https://platform.openai.com/docs/models 
    # Context length = context window. Max token and max length both refer to max output token length. https://community.openai.com/t/context-length-vs-max-token-vs-maximum-length/125585/2 
    # Existing methods in langchain: https://github.com/langchain-ai/langchain/issues/12264
    # Text splitting: https://python.langchain.com/docs/how_to/recursive_text_splitter/
        # chunk size refers to the number of characters per chunk I think: https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846 

    # Handle edge case where the text is a list: TypeError: expected string or bytes-like object, got 'list'
    if isinstance(text, list):
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def execute_with_retry(model, summarizer_model, messages, max_retries=3, delay=2):
    """
    Executes a function with retry logic.

    :param func: The function to execute.
    :param max_retries: The maximum number of retry attempts.
    :param delay: The delay (in seconds) between retries.
    :param kwargs: Arguments to pass to the function.
    :return: The function's result if successful.
    :raises: The last exception encountered if all retries fail.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Case 1: all messages add up to over the context limit: we just prune a set amount of messages in the middle
            # Specifically, we prune 40% of messages among the list of messages, if when adding up all message's individual lengths, the total length exceeds the context limit:
            total_length = sum([len(message) for message in messages])
            if total_length > GPT_4O_MAX_CHARS:
                print(f"Total message length will exceed context limit. Total length of messages now: {total_length}. Pruning messages.")
                # Prune 40% of messages in between the messages list:

                messages_to_prune = messages[len(messages) // 3: -len(messages) // 3]

                # Iterate from the back of messages_to_prune, if we see a ToolMessage, we keep the message that comes before it. If we see any other message, we prune it. 
                j = len(messages_to_prune) - 1
                while j >= 0:
                    if messages_to_prune[j].type == "tool":
                        j -= 2
                    else:
                        messages_to_prune.pop()
                        j -= 1
                        
                messages = messages[:len(messages) // 3] + messages_to_prune + messages[-len(messages) // 3:]


                total_length = sum([len(message) for message in messages])
                print(f"Pruning complete... Total length of messages now: {total_length}.")

            # Case 2: Split the last message into chunks: (since this is the only possibility for exceeding GPT-4o max context length limits)
            splitted_text = text_splitter_gpt_4o(messages[-1].content, GPT_4O_MAX_CHARS)
            # if we did exceed, we will need to iteratively pass the input into the LLM:
            if len(splitted_text) > 1: # my observation is that huge text chunks usually come only from execute_shell_command

                splitted_text = text_splitter_gpt_4o(messages[-1].content, GPT_4O_DEFINED_CHAR_LIMIT)

                summarized_text = ""
                print(f"Last message length will exceed defined context limit. Splitting text into {len(splitted_text)} chunks")
                for i, text_chunk in enumerate(splitted_text):
                    print(f"Processing chunk {i + 1} of {len(splitted_text)}")
                    # Ask our summarizer gpt-4o model to summarize chunks:

                    print("Chunk length before summarize:", len(text_chunk))

                    response = summarizer_model.invoke("Summarize the following text. Be concise, but leave the original formatting/structure intact. A method to do this is to just prune many lines that may not be important. Don't output anything other than the summarized text. Here is the text: \n" + text_chunk)

                    print("Summarizer Response:", response)

                    summarized_text += response.content

                    print("Chunk length after summarize:", len(response.content))

                # Reassign the last message content to the summarized text:
                messages[-1].content = summarized_text

            # Case 3: if the last message exceeds our own defined limit (high leeway), we summarize it:
            splitted_text = text_splitter_gpt_4o(messages[-1].content, GPT_4O_DEFINED_CHAR_LIMIT)
            if len(splitted_text) > 1:
                print("Chunk length before summarize:", len(messages[-1].content))
                print(f"Last message length will exceed defined context limit of {GPT_4O_DEFINED_CHAR_LIMIT}. Summarizing text...")
                messages[-1].content = summarizer_model.invoke("Summarize the following text. Be concise, but leave the original formatting/structure intact. A method to do this is to just prune many lines that may not be important. Don't output anything other than the summarized text. Here is the text: \n" + messages[-1].content).content

                print("Summarizer Response:", messages[-1].content)

                print("Chunk length after summarize:", len(messages[-1].content))

            return model.invoke(messages)

        except BadRequestError as e:
            print(f"Bad request error: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
        except Exception as e: # I expect this will happen when we call text_splitter_gpt_4o with a huge message content
            print(f"Unexpected error: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
    raise RuntimeError(f"Failed after {max_retries} retries.")

def query_model_safe(model, summarizer_model, messages):
    # Messages should be a list of langchain_core.messages objects
    return execute_with_retry(model, summarizer_model, messages)

# print(len(text_splitter_gpt_4o("""
# What I Worked On

# February 2021

# Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

# The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
# """
# )))