
import gzip
import json
import pandas as pd

review_file = "data/All_Beauty.jsonl.gz"
metadata_file = "data/meta_All_Beauty.jsonl.gz"
output_file = "data/real_reviews.csv"

# Read product names from the metadata file
product_names = {}

with gzip.open(metadata_file, "rt", encoding="utf-8") as file:
    for line in file:
        product = json.loads(line)
        product_id = product.get("parent_asin")
        title = product.get("title")

        if product_id and title:
            product_names[product_id] = title

print("Product names loaded:", len(product_names))

# Read a limited number of useful reviews
reviews = []
max_reviews = 2000

with gzip.open(review_file, "rt", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)

        rating = review.get("rating")
        text = review.get("text", "").strip()
        product_id = review.get("parent_asin")

        if not text or rating is None:
            continue

        if rating >= 4:
            sentiment = "positive"
        elif rating <= 2:
            sentiment = "negative"
        else:
            continue

        reviews.append({
            "product_id": product_id,
            "product": product_names.get(
                product_id, "Unknown Product"
            ),
            "review": text,
            "rating": rating,
            "sentiment": sentiment
        })

        if len(reviews) >= max_reviews:
            break

df = pd.DataFrame(reviews)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Reviews saved:", len(df))
print("Saved to:", output_file)
print(df.head())
