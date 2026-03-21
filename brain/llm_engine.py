import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv()

# Get API key
API_KEY = os.getenv("GROQ_API_KEY")

if API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=API_KEY)


def generate_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return content or ""

    except Exception as e:
        return f"Error generating response: {e}"