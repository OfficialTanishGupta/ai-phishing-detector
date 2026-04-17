from flask import Flask, request, jsonify
import torch
import pickle
from model import SpamClassifier
from preprocess import clean_text

app = Flask(__name__)

# Load vectorizer
with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
input_size = 3000
model = SpamClassifier(input_size)
model.load_state_dict(torch.load("../src/spam_model.pth"))
model.eval()


@app.route("/")
def home():
    return "Spam Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    text = clean_text(text)
    vector = vectorizer.transform([text]).toarray()
    vector = torch.tensor(vector).float()

    with torch.no_grad():
        output = model(vector)
        _, predicted = torch.max(output, 1)

    result = "Spam" if predicted.item() == 1 else "Not Spam"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)