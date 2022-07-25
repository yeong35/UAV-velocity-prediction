import os
import librosa
import numpy as np

import torch
from torch.utils.data import Dataset


class AudioUtil():
    def open(audio_file):
        y, sr = librosa.load(audio_file)
        return y, sr
    
    def MFCCs(y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled

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
