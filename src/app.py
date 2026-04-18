from flask import Flask, request, jsonify, render_template
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
model.load_state_dict(torch.load("spam_model.pth"))
model.eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    original_text = request.form["text"]   # ← save original first

    cleaned = clean_text(original_text)    # ← clean separately
    vector = vectorizer.transform([cleaned]).toarray()
    vector = torch.tensor(vector).float()

    with torch.no_grad():
        output = model(vector)
        _, predicted = torch.max(output, 1)

    result = "Spam" if predicted.item() == 1 else "Not Spam"

    return render_template("index.html", prediction=result, input_text=original_text)  # ← pass original


if __name__ == "__main__":
    app.run(debug=True)