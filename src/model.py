import torch.nn as nn

class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
    
