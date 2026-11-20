
import joblib

model = joblib.load("sentiment_model.pkl")

review = input("Enter a product review: ")

prediction = model.predict([review])

print("Predicted sentiment:", prediction[0])
