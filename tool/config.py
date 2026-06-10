"""Tool-use domain constants: system prompt + protocol tokens.

We adopt Qwen3-4B-Thinking-2507's native tool-calling surface form:
  - tools declared as JSON inside `<tools></tools>` in the system message
    (rendered by `apply_chat_template(..., tools=[...])`),
  - calls emitted as `<tool_call>\n{"name":...,"arguments":...}\n</tool_call>`,
  - results returned inside `<tool_response>...</tool_response>` wrapped in a
    user-role turn.

The latent/think reasoning preface lives in the system prompt the same way
the search domain handles its retrieval system prompt.
"""

SYSTEM_PROMPT = (
    "You are a careful tool-using assistant. "
    "Before each action you must reason. You may choose either mode per turn:\n"
    "- <latent>••••</latent> — compact internal reasoning. Emit exactly four "
    "bullet placeholder tokens between the tags; each carries one step of "
    "internal latent state. Use by default for routine steps.\n"
    "- <think> ... </think> — explicit textual reasoning. Use when you need "
    "to chain information from previous tool calls, recover from an "
    "unexpected response, or plan a multi-step sequence.\n"
    "After reasoning, either issue one or more <tool_call> calls, or "
    "produce the final natural-language answer."
)

ACTION_OPEN = "<tool_call>"
ACTION_CLOSE = "</tool_call>"
OBS_OPEN = "<tool_response>"
OBS_CLOSE = "</tool_response>"
