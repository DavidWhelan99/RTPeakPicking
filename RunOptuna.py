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
from optuna.visualization import plot_intermediate_values, plot_param_importances
from RNN import RNN, MaskedBCELoss, check_accuracy, save_checkpoint


def objective(trial):
    num_layers = trial.suggest_int("num_layers", 1, 6)
    hidden_size = 2**trial.suggest_int("hidden_size_exp", 4, 9)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = 2**trial.suggest_int("batch_size_exp", 4, 9)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    print("num_layers=", num_layers, "\n",
           "hidden_size=", hidden_size, "\n",
           "learning_rate=", learning_rate, "\n",
           "batch size=", batch_size, "\n",
           "dropout=", dropout, "\n")

    data_processor = DataProcessor()
    d, test_mzrts = data_processor.process_for_rnn(only_peak=True,
                                                    min_duration=5,
                                                    false_prune_percent=0,
                                                    log_normalize=True,
                                                    min_max_normalize=True,
                                                    smooth=0,
                                                    create_new_json=True,
                                                    add_derivs=False)

    intensities = []
    truths = []
    masks = []
    mzs = []
    rts = []
    #derivs = []

    test_intensities = []
    test_truths = []
    test_masks = []
    test_mzs = []
    test_rts = []
    #test_derivs = []

    for mzml_key in d.keys():
        for mzrt_key in d[mzml_key].keys():
            if mzrt_key in test_mzrts:
                test_intensities.append(d[mzml_key][mzrt_key]["intensity array"])
                test_truths.append(d[mzml_key][mzrt_key]["truth array"])
                test_masks.append(d[mzml_key][mzrt_key]["mask"])
                test_mzs.append(d[mzml_key][mzrt_key]["mz array"])
                test_rts.append(d[mzml_key][mzrt_key]["rt array"])
                #test_derivs.append(d[mzml_key][mzrt_key]["derivative array"])
            else:
                intensities.append(d[mzml_key][mzrt_key]["intensity array"])
                truths.append(d[mzml_key][mzrt_key]["truth array"])
                masks.append(d[mzml_key][mzrt_key]["mask"])
                mzs.append(d[mzml_key][mzrt_key]["mz array"])
                rts.append(d[mzml_key][mzrt_key]["rt array"])
                #derivs.append(d[mzml_key][mzrt_key]["derivative array"])

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_SIZE = 3
    SEQUENCE_LENGTH = 51
    NUM_EPOCHS = 50
    N_TRAIN_EXAMPLES = 10000
    N_VAL_EXAMPLES = 2000

    truths = torch.tensor(truths).unsqueeze(2)
    masks = torch.tensor(masks).unsqueeze(2)
    intensities = torch.tensor(intensities).unsqueeze(2)
    mzs = torch.tensor(mzs).unsqueeze(2)
    rts = torch.tensor(rts).unsqueeze(2)
    #derivs = torch.tensor(derivs).unsqueeze(2)
    intensities = torch.cat((intensities, mzs, rts), 2)

    test_truths = torch.tensor(test_truths).unsqueeze(2)
    test_masks = torch.tensor(test_masks).unsqueeze(2)
    test_intensities = torch.tensor(test_intensities).unsqueeze(2)
    test_mzs = torch.tensor(test_mzs).unsqueeze(2)
    test_rts = torch.tensor(test_rts).unsqueeze(2)
    #test_derivs = torch.tensor(test_derivs).unsqueeze(2)
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

    best_score = 2147483647

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

        if val_score < best_score:
            best_score = val_score
            best_model = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    save_checkpoint(best_model, "SavedWeights/valloss="+str(best_score)+".pth.tar")
    return best_score



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, timeout=54000)

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