import torch
import torch.nn as nn

# Neural Networks example
class ClassifireNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireNN, self).__init__()
        self.fc1 = nn.Linear(2 * 20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):

        x = x.view(-1, 20 * 2)  

        x = self.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.relu(self.fc3(x))
        x = self.drop_out(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        x = torch.sigmoid(x)

        return x.view(-1)

# Not working...
class ClassifireCNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireNN, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        self.pool1 = nn.MaxPool1d(4)
        self.pool2 = nn.MaxPool1d(5)
        self.pool3 = nn.MaxPool1d(5)

        self.fc1 = nn.Linear(2 * 20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        # torch.Size([16, 20, 2]) [batch, feature, channel]

        x = self.relu(self.cnn1(x))
        x = self.pool1(x)
        x = self.relu(self.cnn2(x))
        x = self.pool2(x)
        x = self.relu(self.cnn3(x))      
        x = self.pool3(x)
        
        x = x.view(-1, 20 * 2)  

        x = self.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.relu(self.fc3(x))
        x = self.drop_out(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        x = torch.sigmoid(x)

        return x.view(-1)
