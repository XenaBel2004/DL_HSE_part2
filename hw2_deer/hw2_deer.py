from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

train_size = 500
test_size = 150
k = 5
window = 2
random_seed = 42

random.seed(random_seed)

dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")

label_names = dataset["train"].features["ner_tags"].feature.names

train_data = dataset["train"].select(range(train_size))
test_data = dataset["test"].select(range(test_size))


def tags_to_str(tags):
    return [label_names[t] for t in tags]


def build_deer_statistics(train_data):
    stats = defaultdict(Counter)

    for ex in train_data:
        tokens = ex["tokens"]
        tags = tags_to_str(ex["ner_tags"])

        entity_positions = set()

        for i, tag in enumerate(tags):
            token = tokens[i].lower()

            if tag != "O":
                stats[token]["entity"] += 1
                entity_positions.add(i)

        context_positions = set()

        for pos in entity_positions:
            left = max(0, pos - window)
            right = min(len(tokens), pos + window + 1)

            for j in range(left, right):
                if j not in entity_positions:
                    context_positions.add(j)

        for i, token in enumerate(tokens):
            token = token.lower()

            if i in entity_positions:
                continue
            elif i in context_positions:
                stats[token]["context"] += 1
            else:
                stats[token]["other"] += 1

    deer_scores = {}

    for token, counter in stats.items():
        total = counter["entity"] + counter["context"] + counter["other"]

        p_entity = counter["entity"] / total
        p_context = counter["context"] / total
        p_other = counter["other"] / total

        deer_scores[token] = 2.0 * p_entity + 1.0 * p_context - 0.5 * p_other

    return deer_scores


def deer_retrieve(test_tokens, train_data, deer_scores, k):
    test_tokens = set([t.lower() for t in test_tokens])
    scores = []

    for ex in train_data:
        train_tokens = set([t.lower() for t in ex["tokens"]])
        common_tokens = test_tokens.intersection(train_tokens)

        score = 0

        for token in common_tokens:
            score += deer_scores.get(token, 0)

        scores.append((score, ex))

    scores = sorted(scores, key=lambda x: x[0], reverse=True)

    return [x[1] for x in scores[:k]]


def build_entity_dictionary(examples):
    entity_dict = defaultdict(Counter)

    for ex in examples:
        tokens = ex["tokens"]
        tags = tags_to_str(ex["ner_tags"])

        for token, tag in zip(tokens, tags):
            if tag != "O":
                entity_type = tag.replace("B-", "").replace("I-", "")
                entity_dict[token.lower()][entity_type] += 1

    return entity_dict


def predict_with_dictionary(tokens, entity_dict):
    preds = []

    for token in tokens:
        token_l = token.lower()

        if token_l in entity_dict:
            entity_type = entity_dict[token_l].most_common(1)[0][0]
            preds.append("B-" + entity_type)
        else:
            preds.append("O")

    return preds


deer_scores = build_deer_statistics(train_data)
global_entity_dict = build_entity_dictionary(train_data)

y_true = []
y_baseline = []
y_dictionary = []
y_deer = []

for ex in tqdm(test_data):
    tokens = ex["tokens"]
    true_tags = tags_to_str(ex["ner_tags"])

    y_true.append(true_tags)
    y_baseline.append(["O"] * len(tokens))
    y_dictionary.append(predict_with_dictionary(tokens, global_entity_dict))

    local_examples = deer_retrieve(tokens, train_data, deer_scores, k)
    local_dict = build_entity_dictionary(local_examples)
    y_deer.append(predict_with_dictionary(tokens, local_dict))

results = []

for name, preds in [
    ("baseline", y_baseline),
    ("dictionary_ner", y_dictionary),
    ("deer_modified", y_deer)
]:
    results.append({
        "method": name,
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds)
    })

df = pd.DataFrame(results)

print(df)

df.to_csv("results.csv", index=False)

plt.figure(figsize=(6, 4))
plt.bar(df["method"], df["f1"])
plt.ylabel("F1")
plt.title("F1 comparison")
plt.tight_layout()
plt.savefig("f1_comparison.png")

plt.figure(figsize=(6, 4))

for metric in ["precision", "recall", "f1"]:
    plt.plot(df["method"], df[metric], marker="o", label=metric)

plt.ylabel("score")
plt.title("metrics comparison")
plt.legend()
plt.tight_layout()
plt.savefig("metrics_comparison.png")