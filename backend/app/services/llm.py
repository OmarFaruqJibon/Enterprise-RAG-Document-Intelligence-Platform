import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured")


client = Groq(api_key=api_key)


def ask_llm(question: str) -> str:

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content