import torch
import os
from model_sm_md_lg import ModelSm
from dotenv import load_dotenv
load_dotenv()

SM_PATH = os.getenv('SM_PATH')


#Initializing Model instances
model_types = ['sm']
model_instances = [ModelSm]
model_paths = [SM_PATH]
model_path_dict = {key: value for key, value in zip(model_types, model_paths)}
model_dict = {key: value for key, value in zip(model_types, model_instances)}

'''

Not using this function as achieving accuracy with Sm itself, future work!!

# Defining trainable parameters.
def params_init(model, model_type):
    if model_type == 'lg':
        for param in model.cnn1.parameters():
            param.requires_grad = False
        for param in model.cnn2.parameters():
            param.requires_grad = False
    elif model_type == 'md':
        for param in model.cnn1.parameters():
            param.requires_grad = False
'''

def load_model(model_type):
    model = model_dict[model_type]()
    if os.path.exists(model_path_dict[model_type]):
        model.load_state_dict(torch.load(model_path_dict[model_type]))
    # params_init(model, model_type)
    return model



