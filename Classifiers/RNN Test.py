import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
#extract data from json
torch.manual_seed(1)

with open("Data/RNN_Normalized.json", "r") as f:
    d = json.load(f)

    test_mzrts = d["test mzrts"]

    intensities = []
    truths = []
    masks = []
    mzs = []
    d1s = []
    d2s = []

    count_test = 0
    count_train = 0

    test_intensities = []
    test_truths = []
    test_masks = []
    test_mzs = []
    test_d1s = []
    test_d2s = []
    for mzml_key in d.keys():
        if mzml_key == "test mzrts":
            continue
        for mzrt_key in d[mzml_key].keys():
            if mzrt_key in test_mzrts:
                count_test += 1
                test_intensities.append(d[mzml_key][mzrt_key]["intensity array"])
                test_truths.append(d[mzml_key][mzrt_key]["truth"])
                test_masks.append(d[mzml_key][mzrt_key]["mask"])
                test_mzs.append([d[mzml_key][mzrt_key]["peak m/z"]]*51)
                test_d1s.append(d[mzml_key][mzrt_key]["d1"])
                test_d2s.append(d[mzml_key][mzrt_key]["d2"])
            else:
                count_train += 1
                intensities.append(d[mzml_key][mzrt_key]["intensity array"])
                truths.append(d[mzml_key][mzrt_key]["truth"])
                masks.append(d[mzml_key][mzrt_key]["mask"])
                mzs.append([d[mzml_key][mzrt_key]["peak m/z"]]*51)
                d1s.append(d[mzml_key][mzrt_key]["d1"])
                d2s.append(d[mzml_key][mzrt_key]["d2"])
    f.close()

print(count_train, count_test)
#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Hyperparameters
input_size = 1
sequence_length = 51
num_layers = 2
hidden_size = 32
learning_rate = 0.0003
batch_size = 64
num_epochs = 300



# #load test data
test_truths = torch.tensor(test_truths).unsqueeze(2)
test_masks = torch.tensor(test_masks).unsqueeze(2)
test_intensities = torch.tensor(test_intensities).unsqueeze(2)
test_d1s = torch.tensor(test_d1s).unsqueeze(2)
test_d2s = torch.tensor(test_d2s).unsqueeze(2)
test_mzs = torch.tensor(test_mzs).unsqueeze(2)
#test_intensities = torch.cat((test_intensities, test_d1s, test_d2s), 2)
print(test_intensities[0])
print(test_truths.shape)
print(test_intensities.shape)
print(test_masks.shape)


#load data
truths = torch.tensor(truths).unsqueeze(2)
masks = torch.tensor(masks).unsqueeze(2)
#intensities = (torch.log(torch.tensor(intensities).unsqueeze(2) + 0.0000001))
# d1s = (torch.log(torch.tensor(d1s).unsqueeze(2) + 0.0000001))
# d2s = (torch.log(torch.tensor(d2s).unsqueeze(2) + 0.0000001))
intensities = torch.tensor(intensities).unsqueeze(2)
d1s = torch.tensor(d1s).unsqueeze(2)
d2s = torch.tensor(d2s).unsqueeze(2)
mzs = torch.tensor(mzs).unsqueeze(2)
#intensities = torch.cat((intensities, d1s, d2s), 2)
print(intensities[0])
print(truths.shape)
print(intensities.shape)
print(masks.shape)
# train_size = int(0.8 * len(intensities))
# val_size = len(intensities) - train_size
# train_dataset, val_dataset = random_split(TensorDataset(intensities, truths, masks), [train_size, val_size])

train_dataset = TensorDataset(intensities, truths, masks)
val_dataset = TensorDataset(test_intensities, test_truths, test_masks)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# make RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)



        # forward prop
        out, _ = self.rnn(x, h0)
        #out = out.reshape(out.shape[0], -1)
        #out = self.fc(out[:,-1,:])
        #
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out









def save_checkpoint(state, filename):
    print("saving...")
    torch.save(state, filename)

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()

    def forward(self, scores, targets, masks):
        loss = F.binary_cross_entropy(scores, targets, reduction='none')
        loss = loss * masks
        return loss.sum() / masks.sum()

criterion = MaskedBCELoss()
# loss and optimizer
#criterion = nn.BCELoss()


#acc check
def check_accuracy(loader, model):
    running_loss = 0
    num_samples = 0

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    model.eval()

    with torch.no_grad():
        for x, y, z in loader:
            
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.float32)
            z = z.to(device=device)


            scores = model(x.float())
            predictions = (scores > 0.5).long()
            loss = criterion(scores, y, z)

            running_loss += loss.item()*data.size(0)

            tp_count += (((2*predictions - y) == 1) == z*10-9).sum()
            tn_count += (((2*predictions - y) == 0) == z*10-9).sum()
            fp_count += (((2*predictions - y) == 2) == z*10-9).sum()
            fn_count += (((2*predictions - y) == -1) == z*10-9).sum()
            #print(predictions.sum(), y.sum(), tp_count)

            #print(predictions.shape)
            num_samples += predictions.size(0)
        
            
        #print("Got {} / {} with accuracy {}".format(num_correct, num_samples*64, float(num_correct)/float(num_samples*64)*100))

        print(running_loss/num_samples)
        print("accuracy:", ((tp_count + tn_count)/(tp_count+tn_count+fp_count+fn_count)).item(), "precision:", (tp_count/(tp_count+fp_count)).item(), "recall:", (tp_count/(tp_count + fn_count)).item(), "f1:", (tp_count/(tp_count + (fp_count + fn_count)/2)).item())
        # print("precision:", tp_count/(tp_count+fp_count))
        # print("recall:", tp_count/(tp_count + fn_count))
        # print("f1:", tp_count/(tp_count + (fp_count + fn_count)/2))
        print("tp:", tp_count.item(), "tn:", tn_count.item(), "fp:", fp_count.item(), "fn:", fn_count.item())

    model.train()
    return (running_loss/num_samples, (tp_count/(tp_count + (fp_count + fn_count)/2)).item())

hypers = [(2, 128),(4,128),(2,256),(4,256),(2,512),(4,512)]
#train
for h in hypers:
    print(h)
    num_layers = h[0]
    hidden_size = h[1]
    #init networkd
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs_since_record = 0
    min_loss = 99999
    max_f1 = 0  
    for epoch in range(num_epochs):
        running_loss = 0.0



        for batch_idx, (data, targets, masks) in enumerate(train_loader):
            #cuda if possible
            masks = masks.to(device=device)
            data = data.to(device=device)
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

        print(epoch + 1)

        print("checking train data accuracy...")
        check_accuracy(train_loader, model)

        print("checking val data accuracy...")
        val_loss, f1 = check_accuracy(val_loader, model)

        

        # Assuming you have a list of input feature names or indices
        if epoch%10==0:
            input_feature_labels = ["Input1", "Input2", "Input3", "Input4"]

            # Get the weights for the input to hidden layer
            weights_input_to_hidden = model.rnn.weight_ih_l0.data.cpu().numpy()

            # Plot histograms for each input feature
            for i in range(weights_input_to_hidden.shape[1]):
                plt.hist(weights_input_to_hidden[:, i], bins=50, label=f"{input_feature_labels[i]}", alpha=0.5)

            plt.legend()
            plt.show()

        epochs_since_record += 1
        if val_loss < min_loss:
            epochs_since_record = 0
            min_loss = val_loss
        if f1 > max_f1:
            epochs_since_record = 0
            max_f1 = f1
        
        if epochs_since_record > 30:
            break

        if epochs_since_record == 0:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, "Weights/RNN_normal/RNN_Normalized_"+"_valLoss="+str(val_loss)+".pth.tar")

    # print("checking test data accuracy...")
    # check_accuracy(test_loader, model)

    # print("checking test2 data accuracy...")
    # check_accuracy(test2_loader, model)

    # a = torch.tensor([0.0,0.0,84920.84375,117817.4453125,155771.875,0.0,67398.265625,97914.7109375,0.0,0.0,251842.046875,525901.0,440355.71875,535370.375,359906.3125,391132.0625,302978.46875,181042.890625,153879.25,171810.40625,177353.40625,316559.78125,426396.625,390355.21875,386271.71875,367625.0625,205317.953125,159500.203125,88645.421875,48922.11328125,59018.34375,57775.5390625,24021.169921875])
    # b = torch.tensor([90937.5546875,0.0,0.0,126241.4453125,91355.078125,63763.3828125,71623.9765625,0.0,0.0,0.0,84920.84375,117817.4453125,155771.875,0.0,67398.265625,97914.7109375,0.0,0.0,251842.046875,525901.0,440355.71875,535370.375,359906.3125,391132.0625,302978.46875,181042.890625,153879.25,171810.40625,177353.40625,316559.78125,426396.625,390355.21875,386271.71875,367625.0625,205317.953125,159500.203125,88645.421875,48922.11328125,59018.34375,57775.5390625,24021.169921875,0.0,0.0,53005.2578125,17849.734375,0.0,47513.44921875,0.0,25652.798828125,0,0])
    # test = (a.unsqueeze(1).unsqueeze(0))/483
    # testb = (b.unsqueeze(1).unsqueeze(0))/483
    # #print((test*483/535370.375).flatten())
    # #print((testb*483/535370.375).flatten())
    # model.eval()
    # print(model(test).flatten())
    # print(model(testb).flatten())
    # print((model(test) > 0.5).float().flatten())
    # print((model(testb) > 0.5).float().flatten())
    # model.train()    


