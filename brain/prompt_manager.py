SYSTEM_PROMPT = """
You are a multimodal AI assistant.

You help users understand their environment using speech
and visual information provided as context.

Provide clear and helpful responses.
"""


def build_prompt(context: str, instruction: str) -> str:
    """
    Builds the final prompt sent to the LLM
    """

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Instruction:
{instruction}

Assistant:
"""

    return prompt
