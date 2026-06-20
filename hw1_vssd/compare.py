import os

import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    resnet_result = pd.read_csv(f"{OUT_DIR}/resnet18_result.csv")
    simple_vssd_result = pd.read_csv(f"{OUT_DIR}/vssd_result.csv")

    official_vssd_result = pd.DataFrame([{
        "model": "Official VSSD Micro",
        "test_loss": None,
        "test_acc": 0.12833,
        "test_f1": None,
        "train_time_sec": 3025
    }])

    final_results = pd.concat(
        [resnet_result, simple_vssd_result, official_vssd_result],
        ignore_index=True
    )

    final_results.to_csv(f"{OUT_DIR}/final_results.csv", index=False)

    print(final_results)

    plt.figure()
    plt.bar(final_results["model"], final_results["test_acc"])
    plt.ylabel("test accuracy")
    plt.title("Accuracy comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/comparison_accuracy.png", dpi=200)
    plt.close()

    f1_df = final_results.dropna(subset=["test_f1"])

    plt.figure()
    plt.bar(f1_df["model"], f1_df["test_f1"])
    plt.ylabel("test macro F1")
    plt.title("Macro F1 comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/comparison_f1.png", dpi=200)
    plt.close()

    time_df = final_results.dropna(subset=["train_time_sec"])

    plt.figure()
    plt.bar(time_df["model"], time_df["train_time_sec"] / 60)
    plt.ylabel("training time, min")
    plt.title("Training time comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/comparison_time.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()