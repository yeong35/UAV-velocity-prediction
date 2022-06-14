import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model import ClassifireNN
from preprocess import CustomDataset
from torch.utils.tensorboard import SummaryWriter

# load metadata
metadata = pd.read_csv("./information.csv")

# using GPU
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper parameter
LR = 0.0001
PATIENCE = 3
FACTOR = 0.95
DROP_OUT = 0.3
EPOCHS = 100

# file path
big_fast_path = "./dataset/big_fast/"
big_slow_path = "./dataset/big_slow/"

#preprocess & load dataset
slow_dataset = CustomDataset(big_slow_path, label = 0)
fast_dataset = CustomDataset(big_fast_path, label = 1)

slow_train, slow_valid, slow_test = torch.utils.data.random_split(slow_dataset,
[int(len(slow_dataset)*0.8), int(len(slow_dataset)*0.1), len(slow_dataset) - int(len(slow_dataset) * 0.8) - int(len(slow_dataset) * 0.1)],
generator=torch.Generator().manual_seed(42))

fast_train, fast_valid, fast_test = torch.utils.data.random_split(fast_dataset,
[int(len(fast_dataset)*0.8), int(len(fast_dataset)*0.1), len(fast_dataset) - int(len(fast_dataset) * 0.8) - int(len(fast_dataset) * 0.1)],
generator=torch.Generator().manual_seed(42))

train_dataset = torch.utils.data.ConcatDataset([slow_train, fast_train])
val_dataset = torch.utils.data.ConcatDataset([slow_valid, fast_valid])
test_dataset = torch.utils.data.ConcatDataset([slow_test, fast_test])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)        


model = ClassifireNN(drop_out=DROP_OUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

best_auc = 0
best_epoch = -1
best_pred = []

prev_model = None

wirter = SummaryWriter()

for i in tqdm(range(EPOCHS)):

    # Train
    train_loss = 0
    true_labels = []
    pred_labels = []
    model.train()

    for e_num, (x, y) in enumerate(train_loader):

        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
        
        model.zero_grad()
        pred_y = model(x)

        loss = criterion(pred_y, y)
        train_loss += loss.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    train_auc = accuracy_score(true_labels, pred_labels)

    # Valid
    valid_loss=0
    true_labels=[]
    pred_labels=[]
    model.eval()

    for e_num, (x, y) in enumerate(val_loader):
        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)

        pred_y = model(x)
        loss = criterion(pred_y, y)

        valid_loss += loss.detach()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    valid_auc = accuracy_score(true_labels, pred_labels)
    
    # wirter.add_scalar("")
    print(f"train_loss : {train_loss:.2f} train_auc : {train_auc:.2f} valid_loss : {valid_loss:.2f} valid_auc : {valid_auc:.2f}")

    if valid_auc > best_auc:
        best_pred = pred_labels
        best_auc = valid_auc
        best_epoch = i

        if prev_model is not None:
            os.remove(prev_model)
        prev_model = f'cnn_model_{best_auc}.h5'
        torch.save(model.state_dict(), prev_model)

print(f"best validation acc = {best_auc}, in epoch {best_epoch}")