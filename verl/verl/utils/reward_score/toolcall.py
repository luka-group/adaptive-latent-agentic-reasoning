# Tool-call reward (Tool-N1 paper Eq. 5: binary FormatCorrect ∧ ToolCallMatch).
# Ported from references/Tool-N1/verl/verl/utils/reward_score/toolcall.py
# (compute_score_v0 only — v1-v4 variants not needed for V11 Stage 0).

import json
import random
import re
from collections import Counter


def validate_result(result, answer):
    if len(result) == 0 or len(answer) == 0:
        return 2 if len(result) == len(answer) else 0
    try:
        c1 = Counter((it["name"], json.dumps(it["arguments"], sort_keys=True)) for it in result)
        c2 = Counter((it["name"], json.dumps(it["arguments"], sort_keys=True)) for it in answer)
    except TypeError:
        return 0
    if c1 == c2:
        return 2
    if Counter(it["name"] for it in result) == Counter(it["name"] for it in answer):
        return 1
    return 0


def validate_format(tool_call_list):
    for it in tool_call_list:
        if not isinstance(it, dict):
            return 0
        if "name" not in it or "arguments" not in it:
            return 0
    return 1


def extract_solution_v0(tool_call_str):
    marker = "<|im_start|>assistant"
    index = tool_call_str.rfind(marker)
    if index != -1:
        tool_call_str = tool_call_str[index:]
    output_string = tool_call_str
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = list(re.finditer(pattern, tool_call_str, flags=re.DOTALL))
    if not matches:
        return None, output_string
    last_content = matches[-1].group(1).strip()
    try:
        return json.loads(last_content), output_string
    except json.JSONDecodeError:
        return None, output_string


def compute_score_v0(solution_str, ground_truth, **kwargs):
    """Tool-N1 binary reward. Returns 0 or 1.

    Pass criteria:
      1. response contains both <think> and </think>
      2. response contains <tool_call>[...]</tool_call> with valid JSON
      3. each parsed call has {name, arguments}
      4. (name, arguments) multiset matches ground_truth exactly
    """
    answer = json.loads(ground_truth)
    result, output_string = extract_solution_v0(solution_str)

    do_print = random.randint(1, 64) == 1

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None
    if isinstance(result, dict):
        result = [result]
    if isinstance(answer, str):
        answer = json.loads(answer)

    if do_print:
        print("************toolcall solution_str************")
        print(solution_str)
        print(f"extracted: {result}")
        print(f"answer:    {answer}")

    if result is not None:
        if "<think>" not in output_string or "</think>" not in output_string:
            return 0
    if result is None:
        return 0
    if not validate_format(result):
        return 0
    if validate_result(result, answer) == 2:
        return 1
    return 0


# Alias for the dispatcher
compute_score = compute_score_v0
