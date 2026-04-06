from groq import Groq
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    http_client=httpx.Client(verify=False)
)

def run_llm(prompt, max_tokens=512):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0
    )
    return response.choices[0].message.content