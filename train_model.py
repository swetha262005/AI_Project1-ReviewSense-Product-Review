
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------- LOAD DATASET ----------------
print("Loading customer reviews...")

df = pd.read_csv("data/real_reviews.csv")

required_columns = ["review", "sentiment"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=["review", "sentiment"]).copy()

df["review"] = df["review"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

df = df[df["review"] != ""]

# Keep only the two expected sentiment classes.
df = df[
    df["sentiment"].isin(["positive", "negative"])
].copy()

if df.empty:
    raise ValueError("No valid positive or negative reviews found.")

if df["sentiment"].nunique() < 2:
    raise ValueError(
        "The dataset must contain both positive and negative reviews."
    )

print(f"Total usable reviews: {len(df)}")
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# ---------------- PREPARE DATA ----------------
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining reviews: {len(X_train)}")
print(f"Testing reviews: {len(X_test)}")

# ---------------- CREATE MODEL ----------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# ---------------- TRAIN MODEL ----------------
print("\nTraining the sentiment analysis model...")

model.fit(X_train, y_train)

print("Training completed!")

# ---------------- EVALUATE MODEL ----------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.2f}%")

report = classification_report(
    y_test,
    predictions,
    labels=["negative", "positive"],
    target_names=["Negative", "Positive"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "positive"]
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Explanation:")
print("Rows represent actual sentiment.")
print("Columns represent predicted sentiment.")
print(f"Actual negative reviews: {cm[0].sum()}")
print(f"Actual positive reviews: {cm[1].sum()}")
print(f"Negative reviews correctly identified: {cm[0, 0]}")
print(f"Negative reviews incorrectly predicted positive: {cm[0, 1]}")
print(f"Positive reviews incorrectly predicted negative: {cm[1, 0]}")
print(f"Positive reviews correctly identified: {cm[1, 1]}")

# ---------------- DISPLAY CONFUSION MATRIX ----------------
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

fig, ax = plt.subplots(figsize=(7, 5))

display.plot(
    ax=ax,
    cmap="Purples",
    values_format="d",
    colorbar=False
)

ax.set_title("Sentiment Model - Confusion Matrix")
ax.set_xlabel("Predicted Sentiment")
ax.set_ylabel("Actual Sentiment")

plt.tight_layout()
plt.show()

# ---------------- SAVE EVALUATION REPORT ----------------
with open("model_evaluation.txt", "w", encoding="utf-8") as file:
    file.write("SENTIMENT MODEL EVALUATION\n")
    file.write("=" * 40 + "\n")
    file.write(f"Total reviews: {len(df)}\n")
    file.write(f"Training reviews: {len(X_train)}\n")
    file.write(f"Testing reviews: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write("\n")

print("\nEvaluation report saved as model_evaluation.txt")

# ---------------- SAVE TRAINED MODEL ----------------
joblib.dump(model, "sentiment_model.pkl")

print("\nTrained model saved as sentiment_model.pkl")
print("Evaluation completed successfully!")
