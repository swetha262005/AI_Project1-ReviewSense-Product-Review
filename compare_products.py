
import pandas as pd
import joblib

# Load the trained AI model
model = joblib.load("sentiment_model.pkl")

# Load customer reviews
data = pd.read_csv("data/reviews.csv")

# Predict sentiment for every review
data["predicted_sentiment"] = model.predict(data["review"])

# Compare the products
for product in data["product"].unique():
    product_data = data[data["product"] == product]

    total_reviews = len(product_data)

    positive_count = (
        product_data["predicted_sentiment"] == "positive"
    ).sum()

    negative_count = (
        product_data["predicted_sentiment"] == "negative"
    ).sum()

    positive_percentage = positive_count / total_reviews * 100
    negative_percentage = negative_count / total_reviews * 100

    print("\nProduct:", product)
    print("Total reviews:", total_reviews)
    print("Positive reviews:", positive_count)
    print("Negative reviews:", negative_count)
    print("Positive percentage:", round(positive_percentage, 2), "%")
    print("Negative percentage:", round(negative_percentage, 2), "%")
