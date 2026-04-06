from modules.generator import generate_reasoning
from modules.critic import critique_reasoning
from modules.detector import error_detected
from modules.revisor import revise_reasoning

def run_pipeline(question, domain="general"):
    result = {
        "question": question,
        "initial_reasoning": None,
        "initial_answer": None,
        "critique": None,
        "confidence_score": None,
        "was_revised": False,
        "final_reasoning": None,
        "final_answer": None
    }

    reasoning, answer = generate_reasoning(question, domain=domain)
    result["initial_reasoning"] = reasoning
    result["initial_answer"] = answer

    critique = critique_reasoning(reasoning, domain=domain)
    result["critique"] = critique
    result["confidence_score"] = critique.get("confidence", 0.5)

    if error_detected(critique, domain=domain):
        revised_reasoning, revised_answer = revise_reasoning(
            reasoning,
            critique.get("reason", "unknown error"),
            domain=domain
        )
        result["final_reasoning"] = revised_reasoning
        result["final_answer"] = revised_answer
        result["was_revised"] = True
    else:
        result["final_reasoning"] = reasoning
        result["final_answer"] = answer
        result["was_revised"] = False

    return result