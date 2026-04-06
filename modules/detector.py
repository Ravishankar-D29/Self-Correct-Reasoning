def error_detected(critique, domain="general"):
    score = critique.get("confidence", 0.5)
    error_flag = critique.get("error_found", True)

    if domain == "hotpotqa":
        threshold = 0.85
    elif domain == "strategyqa":
        threshold = 0.75
    else:
        threshold = 0.70

    return score < threshold or error_flag