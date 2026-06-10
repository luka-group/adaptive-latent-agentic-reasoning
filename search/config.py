"""Search-domain constants: system prompt and protocol tokens.

The system prompt tells the model to choose per turn between latent
reasoning (`<latent>••••</latent>`) and verbalized reasoning
(`<think>...</think>`), then issue retrieval actions
(`<search>...</search>`) until enough information is gathered for the
final `<answer>`.
"""

SYSTEM_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning first every time you get new information. "
    "You may choose either mode per turn:\n"
    "- <latent>••••</latent> — compact internal reasoning. Emit exactly "
    "four bullet placeholder tokens between the tags; each carries one "
    "step of internal latent state. Use by default for routine steps.\n"
    "- <think> ... </think> — explicit textual reasoning. Use when you "
    "need to fuse information from multiple searches, when previous "
    "searches were insufficient, or when you need to reflect on your "
    "previous reasoning.\n"
    "After reasoning, if you find you lack some knowledge, you can call "
    "a search engine by <search> query </search> and it will return the "
    "top searched results between <information> and </information>. "
    "You can search as many times as you want. If you find no further "
    "external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)

ACTION_OPEN = "<search>"
ACTION_CLOSE = "</search>"
OBS_OPEN = "<information>"
OBS_CLOSE = "</information>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
