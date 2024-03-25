import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_sm_md_lg import ModelTest


# Loading the data.
filepath = './loaded_datasets/melspectrogram.npy'
with open(filepath, 'rb') as f:
    X_train = np.load(f)
    X_valid = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f)
    y_valid = np.load(f)
    y_test = np.load(f)
    
# Params.
N = X_train.shape[0]
M = X_valid.shape[0]
T = X_test.shape[0]
epochs = 100
weight_decay = 9e-4
lr = [10 ** x for x in list((torch.linspace(-4.5, -5.5, steps=epochs, dtype=torch.float64)))]

batch_size = 32

# Model, optimizer and regularizers.
device = 'cuda'
model_dict = torch.load('./models/model2_checkpoint_epoch_299_accuracy_76.92308044433594.pth')
model = ModelTest()
model.load_state_dict(model_dict['model_state_dict'])
model = model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
l2_reg = lambda model: sum(torch.norm(param)**2 for param in model.parameters())

# Printing trainable params.
print(f'Total Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad == True)}')

# For capturing epoch wise loss
lossi = []
best = 0.0

for itr in range(epochs):

    # Lr update
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr[itr]
    # Shuffling the dataset for epoch.
    shuffle = np.random.permutation(N)
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]

    eacc, eloss = 0.0, 0.0
    steps = epochs // batch_size
    
    for mini in range(steps):
        idx = torch.randint(mini * batch_size, min((mini + 1) * batch_size, N), (batch_size, ))
        X = torch.tensor(X_train[idx, :, :, :],device=device).float()
        y = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        
        # Forward Pass
        logits, probs = model(X)
        preds = torch.argmax(probs, dim = 1)
        accuracy = torch.sum(y == preds)/float(len(y))
        eacc += accuracy

        # Loss
        loss = F.cross_entropy(logits, y) + weight_decay * l2_reg(model)
        eloss += loss.item()

        # Zero Grad
        optimizer.zero_grad()

        # Backward Pass
        loss.backward()

        # Gradient Updation
        optimizer.step()

    with torch.no_grad():
        idv = torch.randint(0, M, (M, ))
        Xv = torch.tensor(X_valid[idv, :, :, :],device=device).float()
        yv = torch.tensor(y_valid[idv], dtype=torch.long, device=device)
        vlogits ,vprobs = model(Xv)
        vloss = F.cross_entropy(vlogits, yv)
        vpreds = torch.argmax(vprobs, dim = 1)
        vaccuracy = torch.sum(yv == vpreds)/float(len(yv)) * 100
        best = max(vaccuracy, best)

    

    eloss = eloss / steps
    eacc = eacc / steps
    lossi.append(eloss)

    if itr % 20 == 0:
        print(f'Epoch: {itr + 1} - Current lr: {lr[itr]:.8f}')
        print(f'Train Loss: {eloss:0.4f} - Train Accuracy: {eacc * 100:0.2f} - Val Loss: {vloss:0.4f} - Val Accuracy: {vaccuracy:0.2f}')


# Saving the model
torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'./models/model2_checkpoint_epoch_{itr}_accuracy_{best}.pth')

# Plotting loss vs. lr
plt.plot(lr, lossi)
plt.xticks(lr)
plt.xlabel('Learning rate')
plt.ylabel('Loss')
plt.savefig('./plots/model2_checkpoint_epoch_{itr}_accuracy_{best}.png')




    



