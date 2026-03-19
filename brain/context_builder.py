def build_context(
    user_input: str,
    vision_data: str = None,
    memory_data: str = None
) -> str:
    """
    Build structured context for the AI assistant
    """

    # Start with user input
    context = f"User request:\n{user_input}"

    # Add vision data if available
    if vision_data:
        context += f"\n\nDetected scene:\n{vision_data}"

    # Add memory data if available
    if memory_data:
        context += f"\n\nRelevant past information:\n{memory_data}"

    return context
