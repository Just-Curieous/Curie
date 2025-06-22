import os
from collections import defaultdict
from typing import List, Dict, Tuple
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
import settings
from model import create_model

# from .base_memory_manager import MemoryManager


class MemoryManager:
    """Task-scoped & plan-scoped message store with pruning + summarization."""

    def __init__(self, max_messages: int = 50):
        self._store: Dict[str, List[BaseMessage]] = defaultdict(list)
        self.max = max_messages   # hard cap as a final safety net
        
        self.log_pruning = getattr(settings, "LOG_PRUNED_MESSAGES", False)
        if self.log_pruning:
            self.pruning_log_file = "/logs/pruning_log.txt"
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.pruning_log_file), exist_ok=True)
            # Clear log file on initialization
            with open(self.pruning_log_file, "w") as f:
                f.write("--- Pruning Log Start ---\n")

    def _log_pruned(self, key: str, original_msgs: List[BaseMessage], pruned_msgs: List[BaseMessage]):
        """Logs the messages that were pruned."""
        if not self.log_pruning:
            return

        original_contents = {msg.content for msg in original_msgs}
        pruned_contents = {msg.content for msg in pruned_msgs}
        removed_contents = original_contents - pruned_contents

        if not removed_contents:
            return

        with open(self.pruning_log_file, "a") as f:
            f.write(f"\n--- Pruning for key: {key} ---\n")
            f.write(f"Original message count: {len(original_msgs)}\n")
            f.write(f"Pruned message count: {len(pruned_msgs)}\n")
            f.write(f"Removed {len(removed_contents)} messages.\n")
            f.write("--- Removed Messages ---\n")
            for content in removed_contents:
                # To get the message type, we have to find it in the original list
                original_msg = next((m for m in original_msgs if m.content == content), None)
                msg_type = type(original_msg).__name__ if original_msg else "Unknown"
                f.write(f"- {msg_type}: {content[:100]}...\n") # Log first 100 chars
            f.write("--- End Pruning for key ---\n")

    # ---------- public API ----------
    def push(self, key: str, *msgs: BaseMessage) -> None:
        self._store[key].extend(msgs)
        self._prune(key)

    def get(self, key: str) -> List[BaseMessage]:
        return self._store[key]

    def replace(self, key: str, msgs: List[BaseMessage]) -> None:
        self._store[key] = msgs

    # ---------- internal helpers ----------
    def _prune(self, key: str) -> None:
        msgs = self._store[key]
        if len(msgs) <= self.max:
            return
        
        original_msgs = list(msgs)
        # Example heuristic: keep first system prompt, last 20 msgs, &
        # a rolling summary every N msgs.
        system = [m for m in msgs if isinstance(m, SystemMessage)][:1]
        tail   = msgs[-20:]
        summary_slot = self._summarize(msgs)
        self._store[key] = system + summary_slot + tail
        
        if self.log_pruning:
            self._log_pruned(key, original_msgs, self._store[key])

    def _summarize(self, msgs: List[BaseMessage]) -> List[BaseMessage]:
        """Very cheap extractive summary; replace with real summarizer if desired."""
        # Pick only human / AI messages that introduced tool calls or
        # produced final artefacts, etc.
        important = [m for m in msgs if getattr(m, "tool_calls", None)]
        return important[-3:]  # last few for context


class CurieMemoryManager(MemoryManager):
    def __init__(self, max_messages: int = 50, summarize_llm=None):
        super().__init__(max_messages)
        self.summarize_llm = summarize_llm or create_model()

    def _summarize(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        # If the last message is a tool message, it's too risky to summarize naively.
        # We might orphan it. Let's just do a simple tail-based prune instead to be safe.
        if messages and isinstance(messages[-1], ToolMessage):
            system = [m for m in messages if isinstance(m, SystemMessage)][:1]
            tail = messages[-10:]
            if system and system[0] in tail:
                return tail
            return system + tail
        
        # Filter out the system message and the last message
        system_message = next((msg for msg in messages if isinstance(msg, SystemMessage)), None)
        if not system_message:
            return messages # Cannot summarize without system prompt

        # Do not summarize if there are not enough messages
        if len(messages) < getattr(settings, "SUMMARIZER_MIN_MESSAGES", 10):
            return messages

        summarization_prompt = [
            system_message,
            HumanMessage(
                content=f"Concisely summarize the following conversation, retaining all key information like decisions, results, and tool calls. This summary will be used as context for a future conversation. The last message is the most recent and should be included as-is, not summarized.\n\nCONVERSATION:\n{get_buffer_string(messages[:-1])}"
            ),
        ]
        
        response = self.summarize_llm.invoke(summarization_prompt)
        summary_message = HumanMessage(content=f"Summary of previous conversation:\n{response.content}")
        
        return [system_message, summary_message, messages[-1]]

    def snapshot(self, key: str) -> Dict:
        """Creates a snapshot of the current memory state for a given key."""
        return {"messages": list(self._store[key])}

    def restore(self, key:str, snapshot: Dict) -> None:
        """Restores the memory state from a snapshot for a given key."""
        self._store[key] = snapshot["messages"]

    def prune_and_summarize(self, key: str) -> None:
        """Prunes the message history for a given key, summarizing if necessary."""
        messages = self._store[key]
        original_messages = list(messages)
        
        if len(messages) > self.max:
            summarized_messages = self._summarize(messages)
            self._store[key] = summarized_messages
        else:
            # Basic pruning if summarization is not needed
            self._prune(key)
        
        if self.log_pruning:
            self._log_pruned(key, original_messages, self._store[key]) 