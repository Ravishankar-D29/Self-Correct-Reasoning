from llm import run_llm

def revise_reasoning(original_reasoning, critique_reason, domain="general"):
    if domain == "strategyqa":
        prompt = f"""The following reasoning has an error.

Original reasoning:
{original_reasoning}

Problem:
{critique_reason}

Rewrite the reasoning correctly.
The last line must be exactly 'Answer: yes' or 'Answer: no'

Corrected reasoning:"""
    else:
        prompt = f"""The following reasoning has an error.

Original reasoning:
{original_reasoning}

Problem:
{critique_reason}

Rewrite the reasoning correctly step by step.
The last line must start with 'Answer:'

Corrected reasoning:"""

    response = run_llm(prompt)

    if "Answer:" in response:
        parts = response.split("Answer:")
        reasoning = parts[0].strip()
        answer = parts[-1].strip().lower()
    else:
        reasoning = response.strip()
        answer = response.strip().lower()

    return reasoning, answer