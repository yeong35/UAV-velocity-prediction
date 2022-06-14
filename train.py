# %%
import numpy as np
import pandas as pd
import sklearn
import torch
import librosa
import librosa.display
import torchaudio
import os
import random

# %%
# load metadata
metadata = pd.read_csv("./information.csv")

metadata.head()

# %%
class AudioUtil():
    def open(audio_file):
        y, sr = torchaudio.load(audio_file)
        return y, sr

    # data augmentation function
    def time_shift(aud, shift_limit):
        y, sr = aud
        _, sig_len = y.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return y.roll(shift_amt), sr
    
    def MFCCs(y, sr):
        y = y.cpu().detach().numpy()
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_scaled = np.mean(mfccs.T, axis=0)
        return mfcc_scaled

# %%
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

# %%
class CustomDataset(Dataset):
    def __init__(self, root, label):
        # file root
        self.root = root
        # slow = 0, fast = 1
        self.label = label

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        # all file path
        self.data_files = [f for f in fs if os.path.isfile(f)]
        self.label = [label] * len(self.data_files)
    
    # __len__
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        y, sr = AudioUtil.open(self.data_files[idx])
        mfcc = AudioUtil.MFCCs(y, sr)
        return mfcc, torch.tensor(self.label[idx])

# %%
# file path
big_fast_path = "./dataset/big_fast/"
big_slow_path = "./dataset/big_slow/"

# %%
slow_dataset = CustomDataset(big_slow_path, label = 0)
fast_dataset = CustomDataset(big_fast_path, label = 1)

slow_train, slow_valid, slow_test = torch.utils.data.random_split(slow_dataset,
[int(len(slow_dataset)*0.8), int(len(slow_dataset)*0.1), len(slow_dataset) - int(len(slow_dataset) * 0.8) - int(len(slow_dataset) * 0.1)],
generator=torch.Generator().manual_seed(42))

fast_train, fast_valid, fast_test = torch.utils.data.random_split(fast_dataset,
[int(len(fast_dataset)*0.8), int(len(fast_dataset)*0.1), len(fast_dataset) - int(len(fast_dataset) * 0.8) - int(len(fast_dataset) * 0.1)],
generator=torch.Generator().manual_seed(42))

# %%
train_dataset = torch.utils.data.ConcatDataset([slow_train, fast_train])
val_dataset = torch.utils.data.ConcatDataset([slow_valid, fast_valid])
test_dataset = torch.utils.data.ConcatDataset([slow_test, fast_test])

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)

# %%
import torch.nn as nn

# %%
# example
class ClassifireNN(nn.Module):
    def __init__(self, drop_out=0.0):
        super(ClassifireNN, self).__init__()
        # self.cnn1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=5, padding=2)
        # self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # self.pool1 = nn.MaxPool1d(4)
        # self.pool2 = nn.MaxPool1d(5)
        # self.pool3 = nn.MaxPool1d(5)

        self.fc1 = nn.Linear(2 * 20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 1)
        
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        # torch.Size([16, 20, 2]) [batch, feature, channel]

        # x = self.relu(self.cnn1(x))
        # # x = self.pool1(x)
        # x = self.relu(self.cnn2(x))
        # # x = self.pool2(x)
        # x = self.relu(self.cnn3(x))      
        # # x = self.pool3(x)
        # print(x.shape)
        # asdf
        
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
        

# %%
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
device

# %%
LR = 0.0001
PATIENCE = 3
FACTOR = 0.95
DROP_OUT = 0.3
EPOCHS = 100

# %%
model = ClassifireNN(drop_out=DROP_OUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# %%
best_auc = 0
best_epoch = -1
best_pred = []

prev_model = None

# %%
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

# %%
wirter = SummaryWriter()

# %%
for i in tqdm(range(EPOCHS)):

    # Train
    loss_sum = 0
    true_labels = []
    pred_labels = []
    model.train()

    for e_num, (x, y) in enumerate(train_loader):

        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
        
        model.zero_grad()
        pred_y = model(x)

        loss = criterion(pred_y, y)
        loss_sum += loss.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    auc = accuracy_score(true_labels, pred_labels)

    # Valid
    for e_num, (x, y) in enumerate(val_loader):
        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

        pred_y = model(x)
        loss = criterion(pred_y, y)

        loss_sum += loss.detach()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    auc = accuracy_score(true_labels, pred_labels)
    
    # wirter.add_scalar("")

    if auc > best_auc:
        best_pred = pred_labels
        best_auc = auc
        best_epoch = i

        if prev_model is not None:
            os.remove(prev_model)
        prev_model = f'cnn_model_{best_auc}.h5'
        torch.save(model.state_dict(), prev_model)

print(f"best validation acc = {best_auc}, in epoch {best_epoch}")

# %%
