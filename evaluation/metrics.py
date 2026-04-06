import pandas as pd

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)

    accuracy = df["correct"].mean() * 100

    error_detection_rate = df["was_revised"].mean() * 100

    revised_df = df[df["was_revised"] == True]
    correction_success = (
        revised_df["correct"].mean() * 100
        if len(revised_df) > 0 else 0.0
    )

    wrong_df = df[df["correct"] == False]
    avg_conf_wrong = (
        wrong_df["confidence_score"].mean()
        if len(wrong_df) > 0 else 0.0
    )

    return {
        "Answer Accuracy":           f"{accuracy:.1f}%",
        "Error Detection Rate":      f"{error_detection_rate:.1f}%",
        "Correction Success Rate":   f"{correction_success:.1f}%",
        "Avg Confidence When Wrong": f"{avg_conf_wrong:.2f}"
    }


if __name__ == "__main__":
    for domain in ["gsm8k", "strategyqa", "hotpotqa"]:
        print(f"\n===== {domain.upper()} METRICS =====")
        try:
            metrics = compute_metrics(f"results/{domain}_results.csv")
            for k, v in metrics.items():
                print(f"  {k:30} {v}")
        except FileNotFoundError:
            print(f"  No results file found for {domain}")