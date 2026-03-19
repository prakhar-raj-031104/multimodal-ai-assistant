from brain.context_builder import build_context
from brain.prompt_manager import build_prompt
from brain.llm_engine import generate_response


def get_mock_inputs():
    """
    Simulate inputs from different modules
    """

    vision_data = "Image shows a mathematical equation on a whiteboard"
    memory_data = "User is learning physics"
    instruction = "Explain clearly in simple terms"

    return vision_data, memory_data, instruction
def main():

    conversation_history = []
    while True:

        user_input = input("User: ")

        # Mock data (temporary)
        vision_data, memory_data, instruction = get_mock_inputs()

        # Build context

        # Add user input to history
        conversation_history.append(f"User: {user_input}")

        # Combine history into a single string
        history_text = "\n".join(conversation_history)

        #Build context using history
        context = build_context(history_text, vision_data, memory_data)

        # Build prompt
        prompt = build_prompt(context, instruction)

        # Generate response
        response = generate_response(prompt)

        print("AI:", response)

        # Store AI response in history
        conversation_history.append(f"AI: {response}")

if __name__ == "__main__":
    main()