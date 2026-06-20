import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"

MAX_LEN = 128
N_SAMPLES = 1200
BATCH_SIZE = 16

os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def load_data():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    texts = dataset["train"]["sentence"][:N_SAMPLES]
    labels = dataset["train"]["label"][:N_SAMPLES]

    return texts, np.array(labels)


def extract_layer_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True
    ).to(device)

    model.eval()

    all_layers = None

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            batch_layer_embeddings = []

            for layer in hidden_states:
                cls_embeddings = layer[:, 0, :].cpu().numpy()
                batch_layer_embeddings.append(cls_embeddings)

            if all_layers is None:
                all_layers = [[] for _ in range(len(hidden_states))]

            for layer_id, emb in enumerate(batch_layer_embeddings):
                all_layers[layer_id].append(emb)

    all_layers = [
        np.vstack(layer_embs)
        for layer_embs in all_layers
    ]

    return all_layers


def evaluate_layers(layer_embeddings, labels):
    results = []

    for layer_id, X in enumerate(layer_embeddings):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels
        )

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({
            "layer": layer_id,
            "accuracy": acc,
            "f1": f1
        })

        print(f"Layer {layer_id}: accuracy={acc:.4f}, f1={f1:.4f}")

    return pd.DataFrame(results)


def plot_results(df):
    plt.figure(figsize=(8, 5))
    plt.plot(df["layer"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["layer"], df["f1"], marker="o", label="F1-score")
    plt.xlabel("Layer number")
    plt.ylabel("Metric value")
    plt.title("Layer informativeness in DistilBERT")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/layer_metrics.png", dpi=300)
    plt.close()


def main():
    texts, labels = load_data()

    print("Extracting embeddings...")
    layer_embeddings = extract_layer_embeddings(texts)

    print("Evaluating layers...")
    df = evaluate_layers(layer_embeddings, labels)

    df.to_csv("results/layer_results.csv", index=False)
    plot_results(df)

    best_layer = df.sort_values("f1", ascending=False).iloc[0]

    print("\nBest layer:")
    print(best_layer)

    print("\nSaved:")
    print("results/layer_results.csv")
    print("results/layer_metrics.png")


if __name__ == "__main__":
    main()