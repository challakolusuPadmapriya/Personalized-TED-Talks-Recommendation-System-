import pandas as pd

df = pd.read_csv("ted_talks.csv")

sample = df.head(500)

sample.to_csv("ted_talks_sample.csv", index=False)

print("Sample dataset created successfully!")
print("Rows:", len(sample))