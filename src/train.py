from preprocess import *
import torch
import pickle
import os
from model import SpamClassifier
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 1. Load and Preprocess Data
df = load_data("../dataset/spam_ham_dataset.csv")
df = preprocess_data(df)

X_data, vectorizer = vectorize_text(df)
y_data = encode_labels(df)

print("Preprocessing Done")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Convert to Tensors
X = torch.tensor(X_train).float()
y = torch.tensor(y_train.values).long()

X_test_t = torch.tensor(X_test).float()
y_test_t = torch.tensor(y_test.values).long()

# 3. Initialize Model
input_size = X.shape[1]
model = SpamClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
for epoch in range(20):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
# 5. Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test_t).sum().item()
    accuracy = correct / y_test_t.size(0)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# --- SAVING SECTION ---

# Ensure the 'models' directory exists
if not os.path.exists("../models"):
    os.makedirs("../models")

# Save the Model weights
torch.save(model.state_dict(), "spam_model.pth")
print("✅ Model saved to spam_model.pth")

# Save the Vectorizer (This was the missing piece!)
with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Vectorizer saved to ../models/vectorizer.pkl")
