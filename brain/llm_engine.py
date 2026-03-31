import os
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize client
client = Groq(api_key=API_KEY)


def generate_response(prompt: str, retries: int = 3) -> str:
    """
    Generate response using Groq LLM with retry mechanism
    """

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ LLM error (attempt {attempt + 1}): {str(e)}")

            # Wait before retry
            time.sleep(1)

    # Final fallback
    return "⚠️ Failed to generate response after multiple attempts."
