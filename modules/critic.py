from llm import run_llm
import json

def critique_reasoning(reasoning, domain="general"):
    if domain == "hotpotqa":
        extra = """Pay special attention to:
- Whether each fact stated is correct
- Whether connections between facts are valid
- Whether the final answer follows from all the hops
Be strict — multi-hop reasoning fails often."""
    elif domain == "strategyqa":
        extra = """Check whether the yes/no conclusion
logically follows from the reasoning steps."""
    else:
        extra = ""

    prompt = f"""You are a reasoning critic.
Review the reasoning below carefully.
{extra}

Return ONLY this JSON object, nothing else:
{{
  "confidence": <float 0.0 to 1.0>,
  "error_found": <true or false>,
  "error_location": "<step number or null>",
  "reason": "<one sentence>"
}}

Reasoning:
{reasoning}

JSON response:"""

    response = run_llm(prompt, max_tokens=200)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except:
        return {
            "confidence": 0.4,
            "error_found": True,
            "error_location": "unknown",
            "reason": "Could not parse critique output"
        }