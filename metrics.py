
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

print("TED TALK RECOMMENDATION SYSTEM")
print("=" * 40)
# Load saved project files
df = pd.read_csv(
    "vectorization_output/talk_metadata.csv"
)

similarity = load_npz(
    "vectorization_output/similarity_matrix.npz"
)

print("Total TED Talks:", len(df))
print("Similarity matrix shape:", similarity.shape)

# Evaluate recommendations for the first 100 talks
top_k = 5
num_talks = min(100, similarity.shape[0])

recommended_indices = set()
scores_list = []
diversity_list = []

for i in range(num_talks):
    scores = similarity.getrow(i).toarray().ravel()

    # Exclude the selected talk itself
    scores[i] = -1

    # Select top 5 recommendations
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_scores = scores[top_indices]

    recommended_indices.update(top_indices.tolist())
    scores_list.extend(top_scores.tolist())

    # Calculate diversity among recommended talks
    submatrix = similarity[top_indices, :]
    submatrix = submatrix[:, top_indices].toarray()

    upper_scores = submatrix[
        np.triu_indices(len(top_indices), k=1)
    ]

    diversity_list.append(1 - np.mean(upper_scores))

# Calculate metrics
coverage = len(recommended_indices) / len(df) * 100
avg_similarity = np.mean(scores_list)
avg_diversity = np.mean(diversity_list)

print("\nEVALUATION RESULTS")
print("-" * 40)
print(f"Catalog coverage: {coverage:.2f}%")
print(f"Average similarity: {avg_similarity:.4f}")
print(f"Average diversity: {avg_diversity:.4f}")

print("\nMetrics evaluation completed successfully!")
