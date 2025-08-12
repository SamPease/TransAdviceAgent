import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=api_key)

def ask_claude(question):
    response = client.messages.create(
        model="claude-3-5-haiku-latest",  
        max_tokens=500,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response.content[0].text

if __name__ == "__main__":
    while True:
        question = input("\nAsk Claude: ")
        if question.lower() in ["quit", "exit"]:
            break
        answer = ask_claude(question)
        print("\nClaude:", answer)