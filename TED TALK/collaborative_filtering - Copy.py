
import pandas as pd
import numpy as np
import pickle

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train_ratings.csv"
MODEL_PATH = BASE_DIR / "collaborative_model.pkl"

# Load training data
ratings = pd.read_csv(TRAIN_PATH)

# Create user and talk mappings
users = sorted(ratings["user_id"].unique())
talks = sorted(ratings["talk_id"].unique())

user_map = {u: i for i, u in enumerate(users)}
talk_map = {t: i for i, t in enumerate(talks)}

# Create user-item rating matrix
rows = ratings["user_id"].map(user_map)
cols = ratings["talk_id"].map(talk_map)

rating_matrix = csr_matrix(
    (
        ratings["rating"].values,
        (rows, cols)
    ),
    shape=(len(users), len(talks))
)

# Train SVD model
n_components = min(
    10,
    len(users) - 1,
    len(talks) - 1
)

if n_components < 1:
    raise ValueError("Not enough users or talks to train SVD.")

model = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = model.fit_transform(rating_matrix)
predicted_ratings = user_factors @ model.components_

# Save model and mappings
model_data = {
    "model": model,
    "user_map": user_map,
    "talk_map": talk_map,
    "users": users,
    "talks": talks,
    "predicted_ratings": predicted_ratings,
    "rating_matrix": rating_matrix
}

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model_data, file)

print("Collaborative filtering trained successfully!")
print("Users:", len(users))
print("Talks:", len(talks))
print("SVD components:", n_components)
print("Model saved:", MODEL_PATH)
