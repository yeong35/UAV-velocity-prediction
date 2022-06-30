from datetime import datetime
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model import ClassifireNN, ClassifireCNN
from preprocess import CustomDataset
from torch.utils.tensorboard import SummaryWriter


# load metadata
metadata = pd.read_csv("./information.csv")

# using GPU
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper parameter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', dest='lr', help='learning rate value', default=0.0001, type=float)
    parser.add_argument('-dropout', dest='dropout', help='drop out', default=0.3, type=float)
    parser.add_argument('-epochs', dest='epochs', help='epochs', default=100, type=int)
    parser.add_argument('-batch', dest='batch', help='batch', default=16, type=int)
    parser.add_argument('-dataset', dest='dataset', help='dataset', default="./dataset", type=str)
    parser.add_argument("--test", dest='test',action="store_true", help="Use model test")
    parser.add_argument('-model_weights', dest='model_weights', help='model path', default='./models/cnn_model_1.0.h5')
    args = parser.parse_args()
    
    return args

args = parse_args()
batch = args.batch
lr = args.lr
dropout = args.dropout
epochs = args.epochs
dataset = args.dataset

SEPARATOR = '======================================='

print(SEPARATOR)
print("Hyperparameter")
print(f"lr             : {lr}")
print(f"drop_out       : {dropout}")
print(f"epochs         : {epochs}")
print(f"batch          : {batch}")
print(f"test           : {args.test}")
print(f"dataset        : {dataset}")
print(SEPARATOR)

# save model path
eventid = f"{datetime.now().strftime('CNN-%Y.%m.%d')}_dropout_{dropout}_lr_{lr}"
output_dir = "./models/" + eventid
os.makedirs(output_dir, exist_ok=True)

# file path
big_fast_path = dataset+"/big_fast_3/"
big_slow_path = dataset+"/big_slow_3/"

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

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(val_dataset, batch_size=batch)


model = ClassifireCNN(drop_out=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

best_auc = 0
best_epoch = -1
best_pred = []

prev_model = None

writer = SummaryWriter(log_dir=output_dir)

if not args.test:

    for i in tqdm(range(epochs)):

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
        
        writer.add_scalar("train_loss", train_loss, global_step=i+1)
        writer.add_scalar("train_auc", train_auc, global_step=i+1)
        writer.add_scalar("valid_loss", valid_loss, global_step=i+1)
        writer.add_scalar("valid_auc", valid_auc, global_step=i+1)

        writer.flush()

        print(f"train_loss : {train_loss:.2f} train_auc : {train_auc:.2f} valid_loss : {valid_loss:.2f} valid_auc : {valid_auc:.2f}")

        if valid_auc > best_auc:
            best_pred = pred_labels
            best_auc = valid_auc
            best_epoch = i

            if prev_model is not None:
                os.remove(prev_model)
            prev_model = output_dir+f'/cnn_model_{best_auc}.h5'
            torch.save(model.state_dict(), prev_model)

    writer.close()
    print(f"best validation acc = {best_auc}, in epoch {best_epoch}")

else:
    # Test
    print("Test trained model")
    model.load_state_dict(torch.load(args.model_weights))

    test_loss = 0
    true_labels = []
    pred_labels = []
    model.eval()

    for e_num, (x, y) in enumerate(test_loader):
        x, y = x.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
        model.zero_grad()
        pred_y = model(x)

        loss = criterion(pred_y, y)

        test_loss += loss.detach()

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(np.around(pred_y.cpu().detach().numpy()))

    test_auc = accuracy_score(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels, target_names=["slow", "fast"]))