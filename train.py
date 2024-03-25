import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from model_loader import load_model
from custom_dataloader import load_data
from model_sm_md_lg import ModelSm
from torch.optim.lr_scheduler import LinearLR
from hubert import model

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

data_list = [X_train, y_train, X_valid, y_valid, X_test, y_test]

for data in data_list:
    data = torch.tensor(data)


parser = argparse.ArgumentParser(description="Train a custom model.")
parser.add_argument("--model_type", type=str, default="sm", help="Type of the model (default: 'sm')")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--cuda", action="store_true", default=True, help="Set CUDA training default True")
args = parser.parse_args()


# Defining trainable parameters.

device = 'cuda'

model = torch.load('./models/lg/distil_2.pth')
#model = model
#model.freeze_feature_extractor()

# for name, param in model.named_parameters():
#     if 'hubert' in name:
#         param.requires_grad = False
model = model.to(device)

print(sum(p.numel() for p in model.parameters() if p.requires_grad == True))

weight_decay = 5e-4
optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
scheduler = LinearLR(optimizer, start_factor=0.1)

# define loss function; CrossEntropyLoss() fairly standard for multiclass problems 
def criterion(predictions, targets): 
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


# define function to create a single step of the training phase
def make_train_step(model, criterion, optimizer):
    
    # define the training step of the training phase
    def train_step(X,Y):
        
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        
        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y)
        
        # Regularization

        l2_reg = torch.tensor(0.).to(device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        loss += weight_decay * l2_reg 
        
        # compute gradients for the optimizer to use 
        loss.backward()
        
        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()
        
        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad() 
        
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model,criterion):
    def validate(X,Y):
        
        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase: 
        with torch.no_grad(): 
            
            # set model to validation phase i.e. turn off dropout and batchnorm layers 
            model.eval()
      
            # get the model's predictions on the validation set
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            
            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits,Y)
            
        return loss.item(), accuracy*100, predictions
    return validate

# get training set size to calculate # iterations and minibatch indices
train_size = X_train.shape[0]

# pick minibatch size (of 32... always)
minibatch = 16

# set device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')

# instantiate model and move to GPU for training
print('Number of trainable params: ',sum(p.numel() for p in model.parameters() if p.requires_grad) )

# instantiate the training step function 
train_step = make_train_step(model, criterion, optimizer=optimizer)

# instantiate the validation loop function
validate = make_validate_fnc(model,criterion)

# instantiate lists to hold scalar performance metrics to plot later
train_losses = []
valid_losses = []

# create training loop for one complete epoch (entire training set)
def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):

    for epoch in range(num_epochs):
        
        # set model to train phase
        model.train()         
        
        # shuffle entire training set in each epoch to randomize minibatch order
        train_indices = np.random.permutation(train_size) 
        
        # shuffle the training set for each epoch:
        X_train = X_train[train_indices] 
        Y_train = Y_train[train_indices]

        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate 
        epoch_acc = 0 
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)
        
        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):
            
            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * minibatch 
            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + minibatch, train_size) 
            # ensure we don't have an index error
            actual_batch_size = batch_end-batch_start 
            
            # get training minibatch with all channnels and 2D feature dims
            X = X_train[batch_start:batch_end] 
            # get training minibatch labels 
            Y = Y_train[batch_start:batch_end] 

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float() 
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            
            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor,Y_tensor) 
            
            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size
            
            # keep track of the iteration to see if the model's too slow
            print('\r'+f'Epoch {epoch}: iteration {i}/{num_iterations}',end='')
        
        with torch.no_grad():
            # create tensors from validation set
            X_valid_tensor = torch.tensor(X_valid,device=device).float()
            Y_valid_tensor = torch.tensor(Y_valid,dtype=torch.long,device=device)
            
            # calculate validation metrics to keep track of progress; don't need predictions now
            valid_loss, valid_acc, _ = validate(X_valid_tensor,Y_valid_tensor)
        
        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        # Decaying the learning rate.
        scheduler.step()
                  
        # # Save checkpoint of the model
        # checkpoint_filename = '/content/gdrive/My Drive/DL/models/checkpoints/parallel_all_you_wantFINAL-{:03d}.pkl'.format(epoch)
        # save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        
        # keep track of each epoch's progress
        print(f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')

# choose number of epochs higher than reasonable so we can manually stop training 
num_epochs = 50
# train it!
train(optimizer, model, num_epochs, X_train, y_train, X_valid, y_valid)

# Save the entire model
torch.save(model, './models/lg/distil_3.pth')


