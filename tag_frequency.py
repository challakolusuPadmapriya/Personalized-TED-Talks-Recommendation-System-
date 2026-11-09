
import csv
import ast
from collections import Counter
import matplotlib.pyplot as plt

file_path = r"D:\TED TALK\cleaned_ted_talks.csv"

tag_counts = Counter()

# Read CSV without pandas
with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        value = row.get("tags", "")

        if not value:
            continue

        try:
            tags = ast.literal_eval(value)

            if isinstance(tags, (list, tuple)):
                tag_counts.update(
                    str(tag).strip()
                    for tag in tags
                    if str(tag).strip()
                )
        except (ValueError, SyntaxError):
            continue

# Select top 20 tags
top_tags = tag_counts.most_common(20)

labels = [tag for tag, count in top_tags][::-1]
counts = [count for tag, count in top_tags][::-1]

# Plot chart
plt.figure(figsize=(11, 8))
plt.barh(labels, counts)
plt.xlabel("Frequency (number of TED talks)")
plt.ylabel("Tag")
plt.title("Top 20 TED Talk Tags by Frequency")
plt.tight_layout()

plt.savefig("ted_tag_frequency.png", dpi=180, bbox_inches="tight")
plt.show()

print("Unique tags:", len(tag_counts))
print("Total tag assignments:", sum(tag_counts.values()))
print("Chart saved as ted_tag_frequency.png")
