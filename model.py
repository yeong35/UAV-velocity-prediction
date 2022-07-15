import torch
import torch.nn as nn

# Neural Networks example
class ClassifireNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireNN, self).__init__()
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 2)
        self.fc5 = nn.Linear(2, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        # x = x.view(-1, 20)  

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

# CNN example
class ClassifireCNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireCNN, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=20, out_channels=8, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(in_channels=4, out_channels= 1, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        # input : [16, 20], [batch, feature]
        x = torch.reshape(x, (-1, 20, 1))   #[batch, feature, 1]
        
        x = self.relu(self.cnn1(x))         # [batch, 8, 1]
        x = self.relu(self.cnn2(x))         # [batch, 4, 1]
        x = self.cnn3(x)                    # [batch, 1, 1]   

        x = torch.sigmoid(x)

        return x.view(-1)
