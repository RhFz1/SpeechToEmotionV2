import torch
import torch.nn as nn

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=128, # input feature (frequency) dim after maxpooling 224*224 -> 224*56 (MFC*time)
            nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dropout=0.4, 
            activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4)
    def forward(self, x):
        x = self.transformer_maxpool(x) # X shape (N, 1, 112, 56)
        x = torch.squeeze(x, dim=1) # Squeezing the channel dimension, retaining only (B,T,F)
        x = x.permute(2, 0, 1) # Reordering the dims as trans accepts (T, B, F)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)
        return x