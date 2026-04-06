from llm import run_llm

def generate_reasoning(question, domain="general"):
    if domain == "strategyqa":
        prompt = f"""Answer the following yes/no question.
Think through it step by step.
The last line must be exactly 'Answer: yes' or 'Answer: no'

Question: {question}"""
    else:
        prompt = f"""Solve the following problem step by step.
Show each reasoning step clearly numbered.
The last line must start with 'Answer:'

Question: {question}"""

    response = run_llm(prompt)

    if "Answer:" in response:
        parts = response.split("Answer:")
        reasoning = parts[0].strip()
        answer = parts[-1].strip().lower()
    else:
        reasoning = response.strip()
        answer = response.strip().lower()

    return reasoning, answer