import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
from DataPreprocessing.DataProcessor import DataProcessor
from optuna.visualization import plot_param_importances
from RNN import RNN, MaskedBCELoss, check_accuracy, save_checkpoint

num_layers = 5
hidden_size = 512
learning_rate = 0.00021442839689367302
batch_size = 32
dropout = 0.5697929897148798
smooth = 0.1

INPUT_SIZE = 3
SEQUENCE_LENGTH = 51
NUM_EPOCHS = 100


data_processor = DataProcessor()
d, test_mzrts = data_processor.process_for_rnn(only_peak=True,
                                                min_duration=5,
                                                false_prune_percent=0,
                                                log_normalize=True,
                                                min_max_normalize=True,
                                                smooth=smooth,
                                                create_new_json=True)

intensities = []
truths = []
masks = []
mzs = []
rts = []

test_intensities = []
test_truths = []
test_masks = []
test_mzs = []
test_rts = []

for mzml_key in d.keys():
    for mzrt_key in d[mzml_key].keys():
        if mzrt_key in test_mzrts:
            test_intensities.append(d[mzml_key][mzrt_key]["intensity array"])
            test_truths.append(d[mzml_key][mzrt_key]["truth array"])
            test_masks.append(d[mzml_key][mzrt_key]["mask"])
            test_mzs.append(d[mzml_key][mzrt_key]["mz array"])
            test_rts.append(d[mzml_key][mzrt_key]["rt array"])
        else:
            intensities.append(d[mzml_key][mzrt_key]["intensity array"])
            truths.append(d[mzml_key][mzrt_key]["truth array"])
            masks.append(d[mzml_key][mzrt_key]["mask"])
            mzs.append(d[mzml_key][mzrt_key]["mz array"])
            rts.append(d[mzml_key][mzrt_key]["rt array"])

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

truths = torch.tensor(truths).unsqueeze(2)
masks = torch.tensor(masks).unsqueeze(2)
intensities = torch.tensor(intensities).unsqueeze(2)
mzs = torch.tensor(mzs).unsqueeze(2)
rts = torch.tensor(rts).unsqueeze(2)
intensities = torch.cat((intensities, mzs, rts), 2)

test_truths = torch.tensor(test_truths).unsqueeze(2)
test_masks = torch.tensor(test_masks).unsqueeze(2)
test_intensities = torch.tensor(test_intensities).unsqueeze(2)
test_mzs = torch.tensor(test_mzs).unsqueeze(2)
test_rts = torch.tensor(test_rts).unsqueeze(2)
test_intensities = torch.cat((test_intensities, test_mzs, test_rts), 2)

train_dataset = TensorDataset(intensities, truths, masks)
val_dataset = TensorDataset(test_intensities, test_truths, test_masks)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

criterion = MaskedBCELoss()
#train
#init networkd
model = RNN(device=device, input_size=INPUT_SIZE, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

max_score = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    print(epoch)

    for batch_idx, (data, targets, masks) in enumerate(train_loader):
        masks = masks.to(device=device)
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        
        #fwd
        scores = model(data)
        loss = criterion(scores, targets, masks)

        running_loss += loss.item()*data.size(0)
        #backward
        optimizer.zero_grad()
        loss.backward()

        #step
        optimizer.step()


    val_score = check_accuracy(val_loader,
                                model,
                                device=device,
                                criterion=criterion,
                                data=data,
                                batch_size=batch_size)


    if val_score > max_score:
        max_score = val_score
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

save_checkpoint(checkpoint, "SavedWeights/5_512_log_minmax_acc="+str(max_score)+".pth.tar")    

