import torch
import torch.nn as nn

class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=20, out_channels=11, kernel_size=5, padding=2, stride=1)
        self.cnn2 = nn.Conv1d(in_channels=11, out_channels=5, kernel_size=5, padding=2, stride=1)
        self.cnn3 = nn.Conv1d(in_channels=5, out_channels= 1, kernel_size=5, padding=2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input : [16, 20], [batch, feature]
        x = torch.reshape(x, (-1, 20, 1))
        
        x = self.relu(self.cnn1(x))         #[batch, 11, 1]
        x = self.relu(self.cnn2(x))         #[batch, 5, 1]
        x = self.cnn3(x)                    #[batch, 1, 1]

        x = torch.sigmoid(x)

        return x.view(-1)