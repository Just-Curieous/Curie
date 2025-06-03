import asyncio
import json
import logging
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from typing import List, cast

import tiktoken
from inspect_ai._util.constants import HTTP
from inspect_ai._util.hooks import send_telemetry
from inspect_ai._util.interrupt import check_sample_interrupt
from inspect_ai._util.trace import trace_action
from inspect_ai._util.working import report_sample_waiting_time, sample_working_time
from inspect_ai.model._cache import CacheEntry, CachePolicy, cache_fetch, cache_store
from inspect_ai.model._call_tools import disable_parallel_tools, tool_call_view, tools_info
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig, active_generate_config
from inspect_ai.model._model import (
    active_model,
    collapse_consecutive_assistant_messages,
    collapse_consecutive_user_messages,
    handle_sample_message_limit,
    record_model_usage,
    resolve_reasoning_history,
    resolve_tool_model_input,
    tool_result_images_as_user_message,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput
from inspect_ai.model._providers.openrouter import OpenRouterError
from inspect_ai.tool import Tool, ToolChoice, ToolFunction, ToolInfo
from inspect_ai.tool._tool_def import ToolDef, tool_defs
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    stop_never,
    wait_exponential_jitter,
)
from tenacity.stop import StopBaseT

logger = logging.getLogger(__name__)


def handle_message_len(
    msg: ChatMessage,
    tokenizer,
    max_tokens: int,
) -> ChatMessage:
    def truncate_string(input_string: str, input_tokens: list, max_tokens: int) -> str:
        n_tokens = len(input_tokens)
        if n_tokens > max_tokens:
            keep_tokens = max_tokens // 2
            first_half = tokenizer.decode(input_tokens[:keep_tokens])
            second_half = tokenizer.decode(input_tokens[-keep_tokens:])
            return first_half + "\n...[content truncated due to length]...\n" + second_half
        return input_string

    if isinstance(msg.content, str):
        item_tokens = tokenizer.encode(msg.content, disallowed_special=())
        msg.content = truncate_string(msg.content, item_tokens, max_tokens)
    elif isinstance(msg.content, list):
        token_lists: list[list[int]] = []  # tokenization of each message
        token_counts: list[int] = []  # number of tokens in each message
        for item in msg.content:
            if item.type == "text":
                item_tokens = tokenizer.encode(item.text, disallowed_special=())
                token_lists.append(item_tokens)
                token_counts.append(len(item_tokens))
            elif item.type == "reasoning":
                item_tokens = tokenizer.encode(item.reasoning, disallowed_special=())
                token_lists.append(item_tokens)
                token_counts.append(len(item_tokens))
            else:
                # Non-text content types don't count towards token limit
                token_lists.append([])
                token_counts.append(0)

        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return msg  # No text content to truncate edge case

        # Distribute max_tokens proportionally for text content
        tokens_per_item = [
            max(1, int((count / total_tokens) * max_tokens)) if count > 0 else 0
            for count in token_counts
        ]

        # Apply truncation while preserving content type
        new_content = []
        for item, item_tokens, max_tokens_for_item in zip(
            msg.content, token_lists, tokens_per_item
        ):
            if item.type == "text":
                item.text = truncate_string(item.text, item_tokens, max_tokens_for_item)
            elif item.type == "reasoning":
                item.reasoning = truncate_string(item.reasoning, item_tokens, max_tokens_for_item)
            new_content.append(item)

        msg.content = new_content

    return msg


def get_gpu_generation() -> str | None:
    """Returns the GPU generation, if available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    generation = result.stdout.strip().split("\n")

    if not generation:
        return None

    return ", ".join([info.strip() for info in generation])


def append_system_message(messages: list[ChatMessage], message: ChatMessageSystem) -> None:
    # find last index of any existing system message
    lastIndex = -1
    for i in list(reversed(range(0, len(messages)))):
        if isinstance(messages[i], ChatMessageSystem):
            lastIndex = i
            break

    # insert it
    messages.insert(lastIndex + 1, message)


def prune_messages(
    messages: List[ChatMessage], prune_individual: bool = False
) -> List[ChatMessage]:
    """Prune messages to stay within API limits.

    Removes older messages while preserving conversation coherence by:
    - Keeping all system messages
    - Keeping the initial task instruction message
    - Removing the oldest 30% of conversation messages
    - Ensuring tool messages remain paired with their parent assistant messages

    Args:
        messages: List of chat messages to prune

    Returns:
        List[ChatMessage]: Pruned list of messages with preserved conversation flow
    """
    # Split messages into system and conversation parts
    system_msgs: List[ChatMessage] = [m for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]

    # Always preserve the first user message (task instructions)
    task_msg = next((m for m in conversation if m.role == "user"), None)

    # Remove oldest 30% of messages
    start_idx = max(1, int(len(conversation) * 0.3))  # Keep 70%

    preserved: List[ChatMessage] = [task_msg] if task_msg else []
    preserved.extend(conversation[start_idx:])
    conversation = preserved

    # OAI API requires any messages with `msg.role == "tool"` to be preceded by
    # an assistant message with corresponding tool_calls. Our pruning may violate this,
    # so we need to clean up tool messages that lost their parent assistant message
    valid_messages = []
    active_tool_ids = set()  # IDs from most recent assistant's tool_calls

    for msg in conversation:
        if "prompt is too long" in msg.content:
            continue
        if msg.role == "assistant":
            # Track any tool calls from this assistant message
            active_tool_ids = {tc.id for tc in (msg.tool_calls or [])}
            valid_messages.append(msg)
        elif msg.role == "tool" and getattr(msg, "tool_call_id", None) in active_tool_ids:
            # Keep tool messages only if they match an active tool call
            valid_messages.append(msg)
        elif msg.role == "user":
            # Reset tool tracking at user messages & keep the message
            active_tool_ids = set()
            valid_messages.append(msg)

    if prune_individual:
        # ensure individual messages are not over the context limit
        MAX_TOKENS_PER_MESSAGE = 190000  # 200k token limit, minus 10k buffer
        # use OAI's 200k tokenizer as an approximation
        tokenizer = tiktoken.get_encoding("o200k_base")

        valid_messages = [
            handle_message_len(msg, tokenizer, MAX_TOKENS_PER_MESSAGE) for msg in valid_messages
        ]

    # Reconstruct pruned conversation
    return cast(List[ChatMessage], system_msgs + valid_messages)


def log_rate_limit_retry(context: str, retry_state: RetryCallState) -> None:
    logger.log(
        HTTP,
        f"{context} rate limit retry {retry_state.attempt_number} after waiting for {retry_state.idle_for}",
    )


async def generate_patched(
    self,
    input: str | list[ChatMessage],
    tools: list[Tool] | list[ToolDef] | list[ToolInfo] | list[Tool | ToolDef | ToolInfo] = [],
    tool_choice: ToolChoice | None = None,
    config: GenerateConfig = GenerateConfig(),
    cache: bool | CachePolicy = False,
) -> ModelOutput:
    """Generate output from the model.

    Args:
        input: Chat message input (if a `str` is passed it is converted
        to a `ChatMessageUser`).
        tools: Tools available for the model to call.
        tool_choice: Directives to the model as to which tools to prefer.
        config: Model configuration.
        cache: Caching behavior for generate responses (defaults to no caching).

    Returns:
        ModelOutput
    """
    # if we are the default model then enforce message limit if it
    # exists (raise an exception if it is exceeded)
    is_active_model = self == active_model()
    if is_active_model:
        handle_sample_message_limit(input)

    # base config for this model
    base_config = self.config

    # if we are the active_model then merge active generate config
    if is_active_model:
        base_config = base_config.merge(active_generate_config())

    # merge passed config
    config = base_config.merge(config)

    # provide max_tokens from the model api if required
    if config.max_tokens is None:
        config.max_tokens = self.api.max_tokens_for_config(config)
        if config.max_tokens is None:
            config.max_tokens = self.api.max_tokens()

    # disable parallel tool calls if requested by any of our tools
    if disable_parallel_tools(tools):
        config.parallel_tool_calls = False

    # normalize input to chat
    if isinstance(input, str):
        input = [ChatMessageUser(content=input)]

    # insert any system message provided in config
    if config.system_message:
        input = [ChatMessageSystem(content=config.system_message)] + input

    # enforce concurrency limits
    start_time = datetime.now()
    working_start = sample_working_time()
    async with self._connection_concurrency(config):
        from inspect_ai.log._samples import track_active_sample_retries

        # generate
        with track_active_sample_retries():
            output = await _generate(
                self=self,
                input=input,
                tools=tools,
                tool_choice=tool_choice,
                config=config,
                cache=cache,
            )

        # update the most recent ModelEvent with the actual start/completed
        # times as well as a computation of working time (events are
        # created _after_ the call to _generate, potentially in response
        # to retries, so they need their timestamp updated so it accurately
        # reflects the full start/end time which we know here)
        from inspect_ai.log._transcript import ModelEvent, transcript

        last_model_event = transcript().find_last_event(ModelEvent)
        if last_model_event:
            last_model_event.timestamp = start_time
            last_model_event.working_start = working_start
            completed = datetime.now()
            last_model_event.completed = completed
            last_model_event.working_time = (
                output.time if output.time is not None else (completed - start_time).total_seconds()
            )

        # return output
        return output


async def _generate(
    self,
    input: list[ChatMessage],
    tools: list[Tool] | list[ToolDef] | list[ToolInfo] | list[Tool | ToolDef | ToolInfo],
    tool_choice: ToolChoice | None,
    config: GenerateConfig,
    cache: bool | CachePolicy = False,
) -> ModelOutput:
    # default to 'auto' for tool_choice (same as underlying model apis)
    tool_choice = tool_choice if tool_choice else "auto"

    # extract tool defs if we can
    tdefs = tool_defs([tool for tool in tools if not isinstance(tool, ToolInfo)])

    # resolve all tools into tool_info
    tools = tools_info(tools)

    # if we have a specific tool selected then filter out the others
    if isinstance(tool_choice, ToolFunction):
        tools = [tool for tool in tools if tool.name == tool_choice.name]

    # if tool_choice is "none" or if there are no tools then fully purge
    # the tools (as some models (e.g. openai and mistral) get confused
    # if you pass them tool definitions along with tool_choice == "none"
    # (they both 'semi' use the tool by placing the arguments in JSON
    # in their output!). on the other hand, anthropic actually errors if
    # there are tools anywhere in the message stream and no tools defined.
    if tool_choice == "none" or len(tools) == 0:
        # allow model providers to implement a tools_required() method to
        # force tools to be passed (we need this for anthropic)
        if not self.api.tools_required():
            tools = []
        tool_choice = "none"

    # handle reasoning history
    input = resolve_reasoning_history(input, config, self.api)

    # apply any tool model_input handlers
    input = resolve_tool_model_input(tdefs, input)

    # break tool image content out into user messages if the model doesn't
    # support tools returning images
    if not self.api.tool_result_images():
        input = tool_result_images_as_user_message(input)

    # optionally collapse *consecutive* messages into one -
    # (some apis e.g. anthropic require this)
    if self.api.collapse_user_messages():
        input = collapse_consecutive_user_messages(input)

    if self.api.collapse_assistant_messages():
        input = collapse_consecutive_assistant_messages(input)

    # retry for transient http errors:
    # - no default timeout or max_retries (try forever)
    # - exponential backoff starting at 3 seconds (will wait 25 minutes
    #   on the 10th retry,then will wait no longer than 30 minutes on
    #   subsequent retries)
    if config.max_retries is not None and config.timeout is not None:
        stop: StopBaseT = stop_after_attempt(config.max_retries) | stop_after_delay(config.timeout)
    elif config.max_retries is not None:
        stop = stop_after_attempt(config.max_retries)
    elif config.timeout is not None:
        stop = stop_after_delay(config.timeout)
    else:
        stop = stop_never

    def before_sleep(retry_state: RetryCallState) -> None:
        wait_time = retry_state.next_action.sleep
        self.total_retry_time += wait_time
        log_rate_limit_retry(self.api.model_name, retry_state)

    @retry(
        wait=wait_exponential_jitter(initial=3, max=(2 * 60), jitter=3),
        retry=retry_if_exception(self.should_retry),
        stop=stop,
        before_sleep=before_sleep,
    )
    async def generate() -> ModelOutput:
        check_sample_interrupt()

        cache_entry: CacheEntry | None
        if cache:
            if isinstance(cache, CachePolicy):
                policy = cache
            else:
                policy = CachePolicy()

            cache_entry = CacheEntry(
                base_url=self.api.base_url,
                config=deepcopy(config),
                input=input,
                model=str(self),
                policy=policy,
                tool_choice=tool_choice,
                tools=tools,  # type: ignore
            )
            existing = cache_fetch(cache_entry)
            if isinstance(existing, ModelOutput):
                self._record_model_interaction(
                    input=input,
                    tools=tools,
                    tool_choice=tool_choice,
                    config=config,
                    cache="read",
                    output=existing,
                    call=None,
                )
                return existing
        else:
            cache_entry = None

        # verify that model apis are allowed
        self.verify_model_apis()

        # record the interaction before the call to generate
        # (we'll update it with the results once we have them)
        complete = self._record_model_interaction(
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
            cache="write" if cache else None,
        )

        with trace_action(logger, "Model", f"generate ({str(self)})"):
            time_start = time.monotonic()
            try:
                # Apply timeout at the API call level if configured
                if config.timeout is not None:
                    async with asyncio.timeout(config.timeout):
                        result = await self.api.generate(
                            input=input,
                            tools=tools,
                            tool_choice=tool_choice,
                            config=config,
                        )
                else:
                    result = await self.api.generate(
                        input=input,
                        tools=tools,
                        tool_choice=tool_choice,
                        config=config,
                    )
            except asyncio.TimeoutError:
                logger.warning(f"API call timed out after {config.timeout} seconds")
                # Create a ModelOutput that indicates the model timed out
                message = ChatMessageAssistant(
                    content="Model exceeded time limit during completion",
                    source="generate",
                )

                output = ModelOutput(
                    model=str(self),
                    choices=[
                        ChatCompletionChoice(
                            message=message,
                            stop_reason="stop",  # Using 'stop' since timeout isn't a valid stop_reason
                        )
                    ],
                    time=config.timeout,  # We know exactly how long it took - the timeout value
                )
                result = (output, None)
            except OpenRouterError as e:
                if (
                    "exceed context limit" in str(e)
                    or "context length" in str(e)
                    or "too long" in str(e)
                ):
                    error_completion = ChatCompletion(
                        choices=[], id="", created=0, model="", object="chat.completion"
                    )
                    error = LengthFinishReasonError(completion=error_completion)

                    # Only add special marker if "too long" is in the error message
                    if "too long" in str(e):
                        error.args = ("PRUNE_INDIVIDUAL_MESSAGES: Message is too long",)

                    raise error
                else:
                    raise e
            finally:
                time_elapsed = time.monotonic() - time_start

        if isinstance(result, tuple):
            output, call = result
        else:
            output = result
            call = None

        # raise error
        if isinstance(output, Exception):
            complete(output, call)

            # Wrap the error in a runtime error which will show the
            # request which caused the error
            error = repr(output)
            request = json.dumps(call.request, indent=2) if call is not None else ""
            error_message = f"{error}\n\nRequest:\n{request}"
            raise RuntimeError(error_message)

        # update output with time (call.time captures time spent
        # on the actual request that succeeds w/ status 200)
        if call and call.time is not None:
            output.time = call.time
        else:
            output.time = time_elapsed

        # add views to tool calls
        for choice in output.choices:
            for tool_call in choice.message.tool_calls or []:
                tool_call.view = tool_call_view(tool_call, tdefs)

        # complete the transcript event
        complete(output, call)

        # record usage
        if output.usage:
            # record usage
            record_model_usage(f"{self}", output.usage)

            # send telemetry if its hooked up
            await send_telemetry(
                "model_usage",
                json.dumps(dict(model=str(self), usage=output.usage.model_dump())),
            )

        if cache and cache_entry:
            cache_store(entry=cache_entry, output=output)

        return output

    # call the model (this will so retries, etc., so report waiting time
    # as elapsed time - actual time for successful model call)
    time_start = time.monotonic()
    model_output = await generate()
    total_time = time.monotonic() - time_start
    if model_output.time:
        report_sample_waiting_time(total_time - model_output.time)

    # return results
    return model_output
