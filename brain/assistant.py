from typing import Optional
from brain.context_builder import build_context
from brain.prompt_manager import build_prompt
from brain.llm_engine import generate_response


def process_user_query(
    user_input: str,
    conversation_history: Optional[str] = None,
    vision_data: Optional[str] = None,
    memory_data: Optional[str] = None,
    instruction: Optional[str] = None
) -> dict:
    """
    Main function to process user query through the AI pipeline
    """

    # Step 1: Build context
    context = build_context(
        user_input=user_input,
        conversation_history=conversation_history,
        vision_data=vision_data,
        memory_data=memory_data
    )

    # Step 2: Build prompt
    prompt = build_prompt(context, instruction)

    # Step 3: Generate response
    response = generate_response(prompt)

    # Step 4: Return structured output (NEW)
    return {
        "response": response
    }