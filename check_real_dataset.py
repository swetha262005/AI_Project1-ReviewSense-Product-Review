
import pandas as pd

df = pd.read_csv("data/real_reviews.csv")

print("Total reviews:", len(df))
print("Unique products:", df["product_id"].nunique())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nTop 10 products by review count:")
print(df["product"].value_counts().head(10))

print("\nMissing values:")
print(df.isnull().sum())
