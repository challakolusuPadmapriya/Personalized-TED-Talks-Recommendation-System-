
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "vectorization_output",
    "talk_metadata.csv"
)

SIMILARITY_PATH = os.path.join(
    PROJECT_DIR,
    "vectorization_output",
    "similarity_matrix.npz"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading TED Talks dataset...")

df = pd.read_csv(METADATA_PATH)

similarity_matrix = load_npz(
    SIMILARITY_PATH
).toarray()

print("Dataset shape:", df.shape)
print("Similarity matrix shape:", similarity_matrix.shape)


# ==========================================
# 3. IDENTIFY REQUIRED COLUMNS
# ==========================================

def find_column(dataframe, possible_names):

    for col in dataframe.columns:

        if col.lower() in possible_names:
            return col

    return None


title_col = find_column(
    df,
    {"title", "name"}
)

tags_col = find_column(
    df,
    {"tags", "tag"}
)

views_col = find_column(
    df,
    {"views", "num_views"}
)

if title_col is None:
    raise ValueError("Title column not found.")

if tags_col is None:
    raise ValueError(
        "Tags column not found in talk_metadata.csv"
    )

if len(df) != similarity_matrix.shape[0]:
    raise ValueError(
        "Dataset and similarity matrix sizes do not match."
    )


# ==========================================
# 4. PREPROCESS TAGS
# ==========================================

def process_tags(value):

    if pd.isna(value):
        return set()

    try:

        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple, set)):
            return {
                str(tag).strip().lower()
                for tag in parsed
                if str(tag).strip()
            }

    except (ValueError, SyntaxError):
        pass

    return {
        tag.strip().lower().strip("'\"")
        for tag in str(value).split(",")
        if tag.strip()
    }


df["processed_tags"] = df[tags_col].apply(process_tags)

print("\nTags preprocessing completed.")


# ==========================================
# 5. CREATE GROUND TRUTH
# ==========================================

# Two talks are considered relevant if
# they share at least one tag.

def get_relevant_talks(index):

    current_tags = df.iloc[index]["processed_tags"]

    if not current_tags:
        return set()

    relevant = set()

    for other_index, other_tags in enumerate(
        df["processed_tags"]
    ):

        if other_index == index:
            continue

        if current_tags.intersection(other_tags):
            relevant.add(other_index)

    return relevant


# ==========================================
# 6. POPULARITY SCORES
# ==========================================

if views_col is not None:

    views = pd.to_numeric(
        df[views_col],
        errors="coerce"
    ).fillna(0).clip(lower=0).to_numpy()

    popularity_scores = np.log1p(views)

    max_score = popularity_scores.max()

    if max_score > 0:
        popularity_scores /= max_score

else:

    popularity_scores = np.zeros(len(df))


# ==========================================
# 7. EVALUATION METRICS
# ==========================================

K = 10


def precision_at_k(recommendations, relevant, k):

    if k == 0:
        return 0.0

    hits = sum(
        1 for item in recommendations[:k]
        if item in relevant
    )

    return hits / k


def recall_at_k(recommendations, relevant, k):

    if len(relevant) == 0:
        return 0.0

    hits = sum(
        1 for item in recommendations[:k]
        if item in relevant
    )

    return hits / len(relevant)


def dcg_at_k(recommendations, relevant, k):

    dcg = 0.0

    for rank, item in enumerate(recommendations[:k]):

        if item in relevant:

            dcg += 1 / np.log2(rank + 2)

    return dcg


def ndcg_at_k(recommendations, relevant, k):

    if len(relevant) == 0:
        return 0.0

    actual_dcg = dcg_at_k(
        recommendations,
        relevant,
        k
    )

    ideal_dcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(min(len(relevant), k))
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def average_precision(recommendations, relevant):

    if len(relevant) == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(recommendations, start=1):

        if item in relevant:

            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


# ==========================================
# 8. GENERATE RECOMMENDATIONS
# ==========================================

def get_recommendations(index, method):

    content_scores = similarity_matrix[index].copy()

    # Remove the input TED Talk itself.
    content_scores[index] = -1

    if method == "Content-Based":

        final_scores = content_scores

    elif method == "Hybrid":

        final_scores = (
            0.8 * content_scores
            + 0.2 * popularity_scores
        )

        final_scores[index] = -1

    else:
        raise ValueError("Unknown recommendation method.")

    recommendations = np.argsort(
        -final_scores
    )

    return recommendations.tolist()


# ==========================================
# 9. EVALUATE BOTH MODELS
# ==========================================

print("\nStarting Module 5 evaluation...")

results = []

methods = [
    "Content-Based",
    "Hybrid"
]

for method in methods:

    print("\nEvaluating:", method)

    for index in range(len(df)):

        relevant = get_relevant_talks(index)

        if not relevant:
            continue

        recommendations = get_recommendations(
            index,
            method
        )

        results.append({

            "Method": method,

            "Talk": df.iloc[index][title_col],

            "Precision@10": precision_at_k(
                recommendations, relevant, K
            ),

            "Recall@10": recall_at_k(
                recommendations, relevant, K
            ),

            "NDCG@10": ndcg_at_k(
                recommendations, relevant, K
            ),

            "MAP": average_precision(
                recommendations, relevant
            )

        })


# ==========================================
# 10. SAVE DETAILED RESULTS
# ==========================================

results_df = pd.DataFrame(results)

detailed_path = os.path.join(
    OUTPUT_DIR,
    "detailed_evaluation.csv"
)

results_df.to_csv(
    detailed_path,
    index=False
)

print("\nDetailed results saved.")


# ==========================================
# 11. CREATE COMPARISON TABLE
# ==========================================

comparison = results_df.groupby(
    "Method"
)[
    [
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "MAP"
    ]
].mean().reset_index()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison.to_string(index=False))


# ==========================================
# 12. VISUALIZE RESULTS
# ==========================================

metrics = [
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "MAP"
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Method"],
        comparison[metric]
    )

    plt.title(metric + " Comparison")
    plt.xlabel("Recommendation Method")
    plt.ylabel("Score")

    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    chart_path = os.path.join(
        OUTPUT_DIR,
        metric.lower().replace("@", "_") + ".png"
    )

    plt.savefig(chart_path)

    plt.close()


# ==========================================
# 13. SAVE EVALUATION REPORT
# ==========================================

report_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write(
        "PERSONALIZED TED TALKS RECOMMENDATION SYSTEM\n"
    )

    file.write("MODULE 5: EVALUATION METRICS\n\n")

    file.write("Evaluation Metrics:\n")

    file.write("1. Precision@10\n")
    file.write("2. Recall@10\n")
    file.write("3. NDCG@10\n")
    file.write("4. Mean Average Precision\n\n")

    file.write("MODEL COMPARISON:\n")

    file.write(
        comparison.to_string(index=False)
    )

    file.write(
        "\n\nRelevance definition: Two talks are relevant "
        "if they share at least one tag.\n"
    )

    file.write(
        "Hybrid model: 80% content similarity and "
        "20% popularity score.\n"
    )

print("\nEvaluation report saved.")


# ==========================================
# 14. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("MODULE 5 COMPLETED SUCCESSFULLY!")
print("===================================")

print("\nOutput folder:", OUTPUT_DIR)

print("\nGenerated files:")
print("1. detailed_evaluation.csv")
print("2. model_comparison.csv")
print("3. evaluation_report.txt")
print("4. precision_10.png")
print("5. recall_10.png")
print("6. ndcg_10.png")
print("7. map.png")

