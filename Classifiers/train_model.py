import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

#extract data from json
with open("Data/RNN_FullRoiMasked_Over5s_PRight.json", "r") as f:
    d = json.load(f)
    intensities = []
    truths = []
    masks = []
    #mzs = []
    d1s = []
    d2s = []
    for mzml_key in d.keys():
        for mzrt_key in d[mzml_key].keys():
            intensities.append(d[mzml_key][mzrt_key]["intensity array"])
            truths.append(d[mzml_key][mzrt_key]["truth"])
            masks.append(d[mzml_key][mzrt_key]["mask"])
            #mzs.append([d[mzml_key][mzrt_key]["peak m/z"]]*2505)
            d1s.append(d[mzml_key][mzrt_key]["d1s"])
            d2s.append(d[mzml_key][mzrt_key]["d2s"])
    f.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 1
sequence_length = 51
num_layers = 2
hidden_size = 32
learning_rate = 0.0003
batch_size = 64
num_epochs = 100

#load data
truths = torch.tensor(truths).unsqueeze(2)
intensities = (torch.log(torch.tensor(intensities).unsqueeze(2) + 0.0000001))
masks = torch.tensor(masks).unsqueeze(2)
print(truths.shape)
print(intensities.shape)
print(masks.shape)
train_size = int(0.8 * len(intensities))
val_size = len(intensities) - train_size
train_dataset, val_dataset = random_split(TensorDataset(intensities, truths, masks), [train_size, val_size])

# train_dataset = TensorDataset(intensities[:train_size], truths[:train_size], masks[:train_size])
# val_dataset = TensorDataset(intensities[train_size:], truths[train_size:], masks[train_size:])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)