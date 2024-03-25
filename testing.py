import torch
import os
import numpy as np
import torch.nn.functional as F


from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = './models/lg/distil.pth' #os.getenv('SM_PATH')
LOADED_DATA_PATH = "./loaded_datasets/waveforms.npy" #os.getenv('LOADED_DATA_PATH')


# model_dict = torch.load(MODEL_PATH)
# model = ModelTest()
# model.load_state_dict(model_dict['model_state_dict'])
model = torch.load(MODEL_PATH)
model = model.to('cuda')

filepath = LOADED_DATA_PATH
    
if os.path.exists(filepath):
# open file in read mode and read data 
    with open(filepath, 'rb') as f:
        X_train = np.load(f)
        X_valid = np.load(f)
        X_test = np.load(f)
        y_train = np.load(f)
        y_valid = np.load(f)
        y_test = np.load(f)

X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
y_test = torch.tensor(y_test, dtype=torch.int64).to('cuda')

with torch.no_grad():
    output_logits, output_softmax = model(X_test)
    predictions = torch.argmax(output_softmax,dim=1)
    accuracy = torch.sum(y_test==predictions)/float(len(y_test))
    # loss = F.cross_entropy(logits, y_test)
    # print(loss.item())
    print(accuracy * 100)

