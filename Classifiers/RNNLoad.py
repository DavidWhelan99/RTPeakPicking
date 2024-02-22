import torch
import torch.nn as nn

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 1
num_layers = 2
hidden_size = 128

# make RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, h0):
        # forward prop
        out, h1 = self.rnn(x, h0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out, h1


#init networkd
model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load("Weights/2_128_GRU.pth.tar", map_location=torch.device(device=device))["state_dict"])

b = torch.tensor([90937.5546875,0.0,0.0,126241.4453125,91355.078125,63763.3828125,71623.9765625,0.0,0.0,0.0,84920.84375,117817.4453125,155771.875,0.0,67398.265625,97914.7109375,0.0,0.0,251842.046875,525901.0,440355.71875,535370.375,359906.3125,391132.0625,302978.46875,181042.890625,153879.25,171810.40625,177353.40625,316559.78125,426396.625,390355.21875,386271.71875,367625.0625,205317.953125,159500.203125,88645.421875,48922.11328125,59018.34375,57775.5390625,24021.169921875,0.0,0.0,53005.2578125,17849.734375,0.0,47513.44921875,0.0,25652.798828125,0,0])
b = torch.log(b + .0000001)
test = b[0:25:5].reshape(5,1,1)
print(test.shape)
model.eval()
print(test.size(0))
h0 = torch.zeros(num_layers, test.size(0), hidden_size).to(device)
c0 = torch.zeros(num_layers, test.size(0), hidden_size).to(device)
x1, h1= model(test, h0)
print("x1",x1)
test2 = b[1:25:5].reshape(5,1,1)
x2, h2 = model(test2, h0)
test3 = b[2:25:5].reshape(5,1,1)
test3 = torch.cat((test3, b[25].reshape(1,1,1)), dim=0)
h2 = torch.cat((h2, torch.zeros(num_layers, 1, hidden_size)), dim=1)
#c2 = torch.cat((c2, torch.zeros(4, 1, 512)), dim=1)
print("x2",x2, h2.shape)
x3, h3 = model(test3, h0)
test4 = b[3:25:5].reshape(5,1,1)
test4 = torch.cat((test4, b[26].reshape(1,1,1)), dim=0)

newtest4 = torch.stack((test4[0], test4[1], test4[2], test4[5], test4[4], test4[3]))
test4 = newtest4
print(h3.shape)
newh3 = torch.stack((h3[:,0,:], h3[:,1,:], h3[:,2,:], h3[:,5,:], h3[:,4,:], h3[:,3,:]), dim=1)
h3 = newh3
print(h3.shape)
#newc3 = torch.stack((c3[:,0,:], c3[:,1,:], c3[:,2,:], c3[:,5,:], c3[:,4,:], c3[:,3,:]), dim=1)
#c3 = newc3

print("x3",x3)
x4, h4 = model(test4, h0)
test5 = b[4:25:5].reshape(5,1,1)
test5 = torch.cat((test5, b[27].reshape(1,1,1)), dim=0)
print("x4",x4)
#print(x4.shape, h4.shape, c4.shape)
#print((model(test) > 0.5).float().flatten())
res, asas = model(b[0:5].reshape(1,5,1), torch.zeros(num_layers, 1, hidden_size).to(device))
print(res.flatten())
tensor1 = torch.tensor([]).reshape(1,0,1)
print(tensor1)
tensor2 = torch.tensor([1, 2, 3, 4, 5]).reshape(1,5,1)
print(tensor2[:,0,:].unsqueeze(1).shape, "fawrtwat")
tensor3 = torch.tensor([6,7,8,9,10]).reshape(1,5,1)
print(tensor2)
tensor4 = torch.cat((tensor1, tensor2, tensor3), dim=1)
print(tensor4.flatten().numpy())


# newb = b.unsqueeze(1).unsqueeze(1)
# out, outh = model(newb, torch.zeros(num_layers, newb.size(0), hidden_size))
# print(out.flatten(), b.flatten())
# print(outh)


model.train()    






# for epoch in range(num_epochs):
#     running_loss = 0.0



#     for batch_idx, (data, targets, masks) in enumerate(train_loader):
#         #cuda if possible
#         masks = masks.to(device=device)
#         data = data.to(device=device)
#         targets = targets.to(device=device, dtype=torch.float32)
        
        
#         #fwd
#         scores = model(data)
#         loss = criterion(scores, targets, masks)

#         running_loss += loss.item()*data.size(0)
#         #backward
#         optimizer.zero_grad()
#         loss.backward()

#         #step
#         optimizer.step()

#     print(epoch + 1)
#     print("checking train data accuracy...")
#     check_accuracy(train_loader, model)

#     print("checking val data accuracy...")
#     val_loss, f1 = check_accuracy(val_loader, model)


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
