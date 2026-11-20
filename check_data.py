
import pandas as pd

# Read the CSV file
data = pd.read_csv("data/reviews.csv")

# Display the first five reviews
print("First five reviews:")
print(data.head())

# Display the number of reviews
print("\nTotal number of reviews:", len(data))

# Display the sentiment counts
print("\nSentiment distribution:")
print(data["sentiment"].value_counts())

# Display the product names
print("\nProducts available:")
print(data["product"].unique())
