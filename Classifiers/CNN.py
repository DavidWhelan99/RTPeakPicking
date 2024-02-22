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
with open("Data/CNN_deriv_32.json", "r") as f:
    d = json.load(f)
    intensities = []
    #mzs = []
    truths = []
    d1s = []
    d2s = []
    print(len(d["intensities"]), len(d["d1s"]), len(d["d2s"]))
    for i in range(len(d["intensities"])):
        intensities.append(d["intensities"][i])
        truths.append(d["truths"][i])
        d1s.append(d["d1s"][i])
        d2s.append(d["d2s"][i])
    f.close()

with open("Data/DATAOnly2_32.json", "r") as f:
    d = json.load(f)
    test_intensities = []
    #mzs = []
    test_truths = []
    for i in range(len(d["intensities"])):
        test_intensities.append(d["intensities"][i])
        test_truths.append(d["truths"][i])
    f.close()

# with open("Data/EIC14_Over5s_IsolatedRois_PRandom.json", "r") as f:
#     d = json.load(f)
#     intensities = []
#     truths = []
#     for mzml_key in d.keys():
#         for mzrt_key in d[mzml_key].keys():
#             intensities.append(d[mzml_key][mzrt_key]["intensity array"])
#             truths.append(d[mzml_key][mzrt_key]["truth"])
#     f.close()


#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 3
sequence_length = 51
num_layers = 2
hidden_size = 32
learning_rate = 0.0003
batch_size = 512
num_epochs = 100


test_truths = torch.tensor(test_truths).unsqueeze(1)
test_intensities = torch.tensor(test_intensities).unsqueeze(1)
print(test_intensities.shape)
print(test_truths.shape)
test_dataset = TensorDataset(test_intensities, test_truths)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



#load data
truths = torch.tensor(truths).unsqueeze(1)
intensities = torch.tensor(intensities).unsqueeze(1)
d1s = torch.tensor(d1s).unsqueeze(1)
d2s = torch.tensor(d2s).unsqueeze(1)
intensities = torch.cat((intensities, d1s, d2s), 1)
print(truths.shape)
print(intensities.shape)

train_size = int(0.8 * len(intensities))
val_size = len(intensities) - train_size
train_dataset, val_dataset = random_split(TensorDataset(intensities, truths), [train_size, val_size])
# train_dataset = TensorDataset(intensities[:train_size], truths[:train_size])
# val_dataset = TensorDataset(intensities[train_size:], truths[train_size:])


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# make RNN
class CNN(nn.Module):
    def __init__(self, kernel_size, padding):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding)
        self.LSTM = nn.LSTM(3, 32, 2, batch_first=True, bidirectional=False)
        self.GRU = nn.GRU(4, 64, 2, batch_first=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3)
        
        self.fc1 = nn.Linear(64, 1)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))

        h0 = torch.zeros(2, x.size(0), 64).to(device)
        # c0 = torch.zeros(2, x.size(0), 32).to(device)

        # # forward prop
        x, _ = self.GRU(x, h0)
        x = self.fc1(x[:,-1,:])
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc1(x)
        x = torch.sigmoid(x)
        return x



def save_checkpoint(state, filename):
    print("saving...")
    torch.save(state, filename)



#init networkd
model = CNN(kernel_size=3, padding=1).to(device)


# loss and optimizer
#criterion = nn.BCELoss(reduction="sum")
criterion = nn.BCELoss(reduction="mean")
eval_criterion = nn.BCELoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#acc check
def check_accuracy(loader, model):
    num_samples = 0
    running_loss = 0
    loss_vals = []

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.float32)


            scores = model(x.float())
            
            predictions = (scores > 0.5).long()

            loss = criterion(scores, y)
            #loss_vals += eval_criterion(scores, y).squeeze().tolist()
            running_loss += loss.item()*x.size(0)

            tp_count += ((2*predictions - y) == 1).sum()
            tn_count += ((2*predictions - y) == 0).sum()
            fp_count += ((2*predictions - y) == 2).sum()
            fn_count += ((2*predictions - y) == -1).sum()
            #print(predictions.sum(), y.sum(), tp_count)

            #print(predictions.shape)
            num_samples += x.size(0)
        
            
        #print("Got {} / {} with accuracy {}".format(num_correct, num_samples*64, float(num_correct)/float(num_samples*64)*100))
        # loss_vals.sort()
        # # filtered_loss_vals = [i for i in loss_vals if abs(i) > 0.1]
        # plt.plot(np.arange(len(loss_vals)), loss_vals)
        # plt.show()
        l = running_loss/num_samples
        print(l)
        print("accuracy:", ((tp_count + tn_count)/(tp_count+tn_count+fp_count+fn_count)).item(), "precision:", (tp_count/(tp_count+fp_count)).item(), "recall:", (tp_count/(tp_count + fn_count)).item(), "f1:", (tp_count/(tp_count + (fp_count + fn_count)/2)).item())
        # print("precision:", tp_count/(tp_count+fp_count))
        # print("recall:", tp_count/(tp_count + fn_count))
        # print("f1:", tp_count/(tp_count + (fp_count + fn_count)/2))
        print("tp:", tp_count.item(), "tn:", tn_count.item(), "fp:", fp_count.item(), "fn:", fn_count.item())
    model.train()
    return l

#train

epochs_since_record = 0
min_loss = 99999
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        #cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device, dtype=torch.float32)
        
        #fwd
        scores = model(data)
        loss = criterion(scores, targets)

        #running_loss += loss.item()

        #backward
        optimizer.zero_grad()
        loss.backward()

        #step
        optimizer.step()
    #print(running_loss/len(train_dataset))
    print(epoch)
    print("checking train data accuracy...")
    check_accuracy(train_loader, model)

    print("checking val data accuracy...")
    val_loss = check_accuracy(val_loader, model)

    # print("checking test data accuracy...")
    # check_accuracy(test_loader, model)

    epochs_since_record += 1
    if val_loss < min_loss:
        epochs_since_record = 0
        min_loss = val_loss
    
    if epochs_since_record > 10:
        break

    if epochs_since_record == 0:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, "Weights/CNN_deriv_32/CNN_deriv_32epoch="+str(epoch)+"_valLoss="+str(val_loss)+".pth.tar")