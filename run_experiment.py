from datasets import load_dataset
import pandas as pd
import os
import json

from pipeline import run_pipeline
from modules.generator import generate_reasoning
from modules.extractor import (
    extract_number,
    extract_gsm8k_ground_truth,
    extract_yesno,
    check_correct
)

os.makedirs("results", exist_ok=True)


# ── dataset loader ────────────────────────────────────────

def load_data(domain, num_questions):
    if domain == "gsm8k":
        return load_dataset("openai/gsm8k", "main",
                            split=f"test[:{num_questions}]")
    elif domain == "strategyqa":
        return load_dataset("ChilleD/StrategyQA",
                            split=f"train[:{num_questions}]")
    else:
        return load_dataset("hotpotqa/hotpot_qa", "distractor",
                            split=f"validation[:{num_questions}]")


# ── ground truth extractor ────────────────────────────────

def get_ground_truth(domain, item):
    if domain == "gsm8k":
        return extract_gsm8k_ground_truth(item["answer"])
    elif domain == "strategyqa":
        raw = item["answer"]
        if isinstance(raw, bool):
            return "yes" if raw else "no"
        gt = str(raw).strip().lower()
        return "yes" if gt == "true" else "no" if gt == "false" else gt
    else:
        return str(item["answer"]).strip().lower()


# ── answer extractor ──────────────────────────────────────

def get_predicted(domain, answer, reasoning=""):
    if domain == "gsm8k":
        return extract_number(answer)
    elif domain == "strategyqa":
        return extract_yesno(answer, reasoning)
    else:
        return str(answer).strip().lower()


# ── correctness check ─────────────────────────────────────

def is_correct(domain, predicted, ground_truth):
    if domain == "gsm8k":
        return check_correct(predicted, ground_truth)
    return predicted == ground_truth


# ── our pipeline ──────────────────────────────────────────

def run_domain(domain, num_questions):
    print(f"\nRunning {domain.upper()} — {num_questions} questions")
    dataset = load_data(domain, num_questions)
    rows, traces = [], []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = get_ground_truth(domain, item)
        print(f"  Q{i+1}/{num_questions}...", end=" ")

        try:
            result = run_pipeline(question, domain=domain)
        except Exception as e:
            print("ERROR:", e)
            continue

        predicted = get_predicted(
            domain,
            result["final_answer"],
            result["final_reasoning"]
        )
        correct = is_correct(domain, predicted, ground_truth)

        if domain == "strategyqa":
            print(
                "revised" if result["was_revised"] else "accepted",
                "| correct" if correct else "| wrong",
                f"| gt:{ground_truth} pred:{predicted}"
            )
        else:
            print(
                "revised" if result["was_revised"] else "accepted",
                "| correct" if correct else "| wrong"
            )

        row = {
            "domain":           domain,
            "question":         question[:80] + "..." if len(question) > 80 else question,
            "ground_truth":     ground_truth,
            "predicted":        predicted,
            "correct":          correct,
            "was_revised":      result["was_revised"],
            "confidence_score": result["confidence_score"],
            "critique_reason":  result["critique"].get("reason", ""),
            "error_location":   result["critique"].get("error_location", "null")
        }
        rows.append(row)
        traces.append({
            **row,
            "initial_answer":    result["initial_answer"],
            "final_answer":      result["final_answer"],
            "initial_reasoning": result["initial_reasoning"],
            "final_reasoning":   result["final_reasoning"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{domain}_results.csv", index=False)
    with open(f"results/{domain}_traces.json", "w") as f:
        json.dump(traces, f, indent=2)
    print(f"Saved {domain}_results.csv + {domain}_traces.json")
    return df


# ── baseline ──────────────────────────────────────────────

def run_baseline(domain, num_questions):
    print(f"\nRunning BASELINE {domain.upper()} — {num_questions} questions")
    dataset = load_data(domain, num_questions)
    rows = []

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = get_ground_truth(domain, item)
        print(f"  Q{i+1}/{num_questions}...", end=" ")

        try:
            reasoning, answer = generate_reasoning(question, domain=domain)
        except Exception as e:
            print("ERROR:", e)
            continue

        predicted = get_predicted(domain, answer, reasoning)
        correct = is_correct(domain, predicted, ground_truth)
        print("correct" if correct else "wrong")

        rows.append({
            "domain":       domain,
            "question":     question[:80] + "..." if len(question) > 80 else question,
            "ground_truth": ground_truth,
            "predicted":    predicted,
            "correct":      correct
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{domain}_baseline.csv", index=False)
    print(f"Saved {domain}_baseline.csv")
    return df


# ── main ──────────────────────────────────────────────────

if __name__ == "__main__":

    NUM_QUESTIONS = 10  # change to 200 for Review 3
    DOMAINS = ["gsm8k", "strategyqa", "hotpotqa"]

    # our pipeline
    our_results = {d: run_domain(d, NUM_QUESTIONS) for d in DOMAINS}

    # baselines
    base_results = {d: run_baseline(d, NUM_QUESTIONS) for d in DOMAINS}

    # combine and save
    df_all = pd.concat(our_results.values(), ignore_index=True)
    df_base = pd.concat(base_results.values(), ignore_index=True)
    df_all.to_csv("results/all_results.csv", index=False)
    df_base.to_csv("results/all_baseline.csv", index=False)

    # print report
    if df_all.empty or df_base.empty:
        print("\nNo results — check API token limit.")
    else:
        print("\n" + "=" * 57)
        print(f"{'Domain':15} | {'Baseline':10} | {'Ours':10} | {'Gain':8}")
        print("=" * 57)

        for d in DOMAINS:
            our_acc  = our_results[d]["correct"].mean() * 100 if len(our_results[d]) > 0 else 0
            base_acc = base_results[d]["correct"].mean() * 100 if len(base_results[d]) > 0 else 0
            gain = our_acc - base_acc
            print(f"{d:15} | {base_acc:8.1f}%  | {our_acc:8.1f}%  | {'+' if gain >= 0 else ''}{gain:.1f}%")

        print("=" * 57)
        print("\nDetailed:")
        for d in DOMAINS:
            df = our_results[d]
            if len(df) == 0:
                continue
            print(
                f"  {d:15} | "
                f"accuracy: {df['correct'].mean()*100:.1f}% | "
                f"revised: {df['was_revised'].mean()*100:.1f}% | "
                f"avg confidence: {df['confidence_score'].mean():.2f}"
            )

        print("\nFiles saved in results/")