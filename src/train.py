from preprocess import *
import torch
from model import SpamClassifier

df = load_data("../dataset/spam_ham_dataset.csv")


df = preprocess_data(df)

X, vectorizer = vectorize_text(df)
y = encode_labels(df)

print("Preprocessing Done")
# print("Shape of X:", X.shape)
# print("Sample y:", y[:5])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X = torch.tensor(X_train).float()
y = torch.tensor(y_train.values).long()

X_test_t = torch.tensor(X_test).float()
y_test_t = torch.tensor(y_test.values).long()

input_size = X.shape[1]
model = SpamClassifier(input_size)

import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    
    
model.eval()

with torch.no_grad():
    # Get predictions for the test set
    test_outputs = model(X_test_t)
    # The highest value in the output is our predicted class (0 or 1)
    _, predicted = torch.max(test_outputs, 1)
    
    # Calculate accuracy
    correct = (predicted == y_test_t).sum().item()
    accuracy = correct / y_test_t.size(0)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), "spam_model.pth")
print("Model saved to spam_model.pth")
