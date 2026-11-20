
import gzip
import json

file_path = "data/All_Beauty.jsonl.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as file:
    for i, line in enumerate(file):
        review = json.loads(line)
        print("Review number:", i + 1)
        print("Available columns:", list(review.keys()))
        print("Product ID:", review.get("parent_asin"))
        print("Rating:", review.get("rating"))
        print("Review:", review.get("text"))
        print("-" * 40)

        if i == 2:
            break
