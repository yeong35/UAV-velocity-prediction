import torch
import torch.nn as nn

# Neural Networks example
class ClassifireNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireNN, self).__init__()
        self.fc1 = nn.Linear(431 * 20, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = x.view(-1, 20 * 431)  

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
        super(ClassifireCNN, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        self.pool1 = nn.MaxPool1d(4)
        self.pool2 = nn.MaxPool1d(5)
        self.pool3 = nn.MaxPool1d(5)

        self.fc1 = nn.Linear(4 * 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        # input shape : [16, 20, 431] [batch, feature, channel]

        x = self.relu(self.cnn1(x)) # [batch, 32, 431]
        x = self.pool1(x)           # [batch, 32, 107]
        x = self.relu(self.cnn2(x)) # [batch, 64, 107]
        x = self.pool2(x)           # [batch, 64, 21]
        x = self.relu(self.cnn3(x)) # [batch, 128, 21]   
        x = self.pool3(x)           # [batch, 128, 4]

        x = x.view(-1, 128 * 4)  

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
