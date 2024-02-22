import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
from DataPreprocessing.DataProcessor import DataProcessor
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import sklearn
from optuna.visualization import plot_param_importances

class RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

        # Weight initialization for GRU layer
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param, nonlinearity="relu")
        
        # Weight initialization for Linear layer
        init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # forward prop
        out, _ = self.rnn(x, h0)
        #out = out.reshape(out.shape[0], -1)
        #out = self.fc(out[:,-1,:])
        out = self.dropout(out)
        out = F.relu(out)
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

#acc check
def check_accuracy(loader, model, device, criterion, data, batch_size, max_examples=float("inf")):
    running_loss = 0
    num_samples = 0

    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y, z) in enumerate(loader):
            if batch_idx*batch_size > max_examples:
                break
            x = x.to(device=device)
            y = y.to(device=device, dtype=torch.float32)
            z = z.to(device=device)

            scores = model(x.float())
            predictions = (scores > 0.5).long()
            loss = criterion(scores, y, z)

            running_loss += loss.item()*x.size(0)

            tp_count += (((2*predictions - torch.round(y)) == 1) == z*10-9).sum()
            tn_count += (((2*predictions - torch.round(y)) == 0) == z*10-9).sum()
            fp_count += (((2*predictions - torch.round(y)) == 2) == z*10-9).sum()
            fn_count += (((2*predictions - torch.round(y)) == -1) == z*10-9).sum()

            num_samples += x.size(0)
        
        accuracy = ((tp_count + tn_count)/(tp_count+tn_count+fp_count+fn_count)).item()
        precision = (tp_count/(tp_count+fp_count)).item()
        recall = (tp_count/(tp_count + fn_count)).item()
        f1 = (tp_count/(tp_count + (fp_count + fn_count)/2)).item()
        

        print(running_loss/num_samples)
        print("acc:", accuracy, "prec:", precision, "recall:", recall, "f1:", f1)
        #print("tp:", tp_count.item(), "tn:", tn_count.item(), "fp:", fp_count.item(), "fn:", fn_count.item())

    model.train()
    return running_loss/num_samples

def objective(trial):
    num_layers = trial.suggest_int("num_layers", 1, 8)
    hidden_size = 2**trial.suggest_int("hidden_size_exp", 1, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = 2**trial.suggest_int("batch_size_exp", 5, 10)
    dropout = trial.suggest_float("dropout", 0.1, 0.8)
    # only_peak = trial.suggest_categorical("only_peak", [True, False])
    # min_duration = trial.suggest_int("min_duration", 0, 10)
    # false_prune_percent = trial.suggest_int("false_prune_percent", 0, 80)
    # log_normalize = trial.suggest_categorical("log_normalize", [True, False])
    # min_max_normalize = trial.suggest_categorical("min_max_normalize", [True, False])
    smooth = trial.suggest_int("smooth", 0, 10)/50

    print("num_layers=", num_layers, "\n",
           "hidden_size=", hidden_size, "\n",
           "learning_rate=", learning_rate, "\n",
           "batch size=", batch_size, "\n",
           "dropout=", dropout, "\n",
        #    "only_peak=", only_peak, "\n",
        #    "log_normalize=", log_normalize, "\n",
        #    "min_max_normalize=", min_max_normalize, "\n",
           "smooth=", smooth)

    data_processor = DataProcessor()
    d, test_mzrts, max_vals = DataProcessor.process_for_rnn(data_processor,
                                                             only_peak=True,
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

    INPUT_SIZE = 3
    SEQUENCE_LENGTH = 51
    NUM_EPOCHS = 100
    N_TRAIN_EXAMPLES = 20000
    N_VAL_EXAMPLES = 4000

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

        for batch_idx, (data, targets, masks) in enumerate(train_loader):
            if batch_idx*batch_size > N_TRAIN_EXAMPLES:
                break
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


        # print("checking train data f1...")
        # train_f1 = check_accuracy(train_loader, model, device=device, criterion=criterion, data=data)

        #print("checking val data f1...")
        val_score = check_accuracy(val_loader,
                                 model,
                                 device=device,
                                 criterion=criterion,
                                 data=data,
                                 batch_size=batch_size,
                                 max_examples=N_VAL_EXAMPLES)

        trial.report(val_score, epoch)

        print(epoch, val_score)

        if val_score > max_score:
            max_score = val_score

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return max_score



if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, timeout=36000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig = plot_param_importances(study)
    fig.show()