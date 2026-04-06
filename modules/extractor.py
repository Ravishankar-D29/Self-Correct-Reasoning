import re

def extract_number(text):
    if text is None:
        return None
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return numbers[-1] if numbers else None

def extract_gsm8k_ground_truth(answer_text):
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        return match.group(1)
    return extract_number(answer_text)

def check_correct(predicted, ground_truth):
    if predicted is None or ground_truth is None:
        return False
    return str(predicted).strip() == str(ground_truth).strip()

def extract_yesno(text, reasoning=""):
    t = str(text).strip().lower()
    if t.startswith("yes"):
        return "yes"
    elif t.startswith("no"):
        return "no"
    words = str(reasoning).strip().lower().split()[:30]
    if "yes" in words:
        return "yes"
    elif "no" in words:
        return "no"
    return "unknown"