import torch
import pickle
import os
from model import SpamClassifier
from preprocess import clean_text

# --- AUTOMATIC PATH LOCATOR ---
# This finds exactly where your predict.py file is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# We define the three most likely places for your vectorizer
possible_paths = [
    os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl"), # Outside src in a models folder
    os.path.join(BASE_DIR, "models", "vectorizer.pkl"),      # Inside src in a models folder
    os.path.join(BASE_DIR, "vectorizer.pkl")                 # Right next to predict.py
]

vectorizer_path = None
for path in possible_paths:
    if os.path.exists(path):
        vectorizer_path = path
        break

if vectorizer_path is None:
    print("❌ ERROR: Could not find 'vectorizer.pkl' anywhere!")
    print(f"Checked: {possible_paths}")
    exit()

# 1. Load the vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)
print(f"✅ Success: Loaded vectorizer from {vectorizer_path}")

# 2. Setup the model with your exact path
model_path = r"C:\Users\Tanish_Gupta\ai-phishing-detector\src\spam_model.pth"
input_size = 3000 
model = SpamClassifier(input_size)

try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ Success: Loaded model from {model_path}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit()

def predict_email(text):
    # Clean text
    text = clean_text(text)
    
    # Convert to vector
    vector = vectorizer.transform([text]).toarray()
    vector = torch.tensor(vector).float()
    
    # Predict
    with torch.no_grad():
        output = model(vector)
        _, predicted = torch.max(output, 1)
    
    return "Spam" if predicted.item() == 1 else "Not Spam"

# Test manually
if __name__ == "__main__":
    print("-" * 30)
    sample = input("Enter email text to check: ")
    result = predict_email(sample)
    print(f"PREDICTION: {result}")
    print("-" * 30)
