import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assistant import Assistant

if __name__ == "__main__":
    assistant = Assistant()

    user_input = input("Enter your query: ")

    result = assistant.run(user_input)

    print("\n🧠 FINAL RESPONSE:\n")
    print(result["response"])   