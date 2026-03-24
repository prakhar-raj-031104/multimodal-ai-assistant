from brain.assistant import process_user_query


def get_mock_inputs():
    """
    Simulate inputs from different modules.
    """

    vision_data = "Image shows a mathematical equation on a whiteboard"
    audio_data = "The speaker says Newton's second law explains force and motion."
    memory_data = "User is learning physics"
    instruction = "Explain clearly in simple terms"

    return vision_data, audio_data, memory_data, instruction


def main():
    conversation_history = []

    while True:
        user_input = input("User: ")

        # Mock inputs (clean way)
        vision_data, audio_data, memory_data, instruction = get_mock_inputs()

        # Convert history list -> string
        history_text = "\n".join(conversation_history)

        # Call the LLM pipeline
        response = process_user_query(
            user_input=user_input,
            conversation_history=history_text,
            vision_data=vision_data,
            audio_data=audio_data,
            memory_data=memory_data,
            instruction=instruction,
        )

        print("AI:", response["response"])

        # Update history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"AI: {response['response']}")


if __name__ == "__main__":
    main()
