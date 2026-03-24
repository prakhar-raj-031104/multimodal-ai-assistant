from typing import Optional
SYSTEM_PROMPT = """
You are a helpful multimodal AI assistant.

You are given context from different sources such as conversation history, visual input, audio input, and memory.

Your job is to:
- Understand the user's request clearly
- Use the provided context to give accurate answers
- Explain in a simple and clear way
- Use step-by-step explanation when needed
- Avoid making up information if unsure

Always respond in a helpful and structured manner.
"""


def build_prompt(context: str, instruction: Optional[str]) -> str:
    """
    Builds the final prompt sent to the LLM
    """

    # Safety checks
    if not context or context.strip() == "":
        context = "No context provided."

    if not instruction or not instruction.strip():
        instruction = "Provide a clear and helpful response."

    instruction_lower = instruction.lower()
    response_rules = []

    if "brief" in instruction_lower:
        response_rules.append("Give a short and concise answer.")

    if "step" in instruction_lower:
        response_rules.append("Explain step-by-step.")

    if "detail" in instruction_lower:
        response_rules.append("Provide a detailed explanation.")

    if not response_rules:
        response_rules.append("Provide a clear and structured answer.")

    response_format = "\n".join(response_rules)

    prompt = f"""
{SYSTEM_PROMPT}

--- CONTEXT ---
{context}

--- USER INSTRUCTION ---
{instruction}

--- RESPONSE FORMAT ---
{response_format}

Assistant:
"""

    return prompt