import torch
import torch.nn as nn
from models_sharded import Trans

class ModelSm(nn.Module):
    def __init__(self):
        super(ModelSm, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3 , 3)), # (B, 32, 126, 280)
            nn.MaxPool2d(kernel_size=(4 , 4)), # (B, 32, 31, 70)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3 , 3)), # (B, 32, 29, 68)
            nn.MaxPool2d(kernel_size=(2 , 4)), # (B, 32, 14, 17)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3 , 3)), # (B, 32, 12, 15)
            nn.MaxPool2d(kernel_size=(4 , 4)), # (B, 128, 3, 3)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=(3 , 3)), # (B, 32, 1, 1)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3 , 3)), # (B, 32, 126, 280)
            nn.MaxPool2d(kernel_size=(4 , 4)), # (B, 32, 31, 70)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3 , 3)), # (B, 32, 29, 68)
            nn.MaxPool2d(kernel_size=(2 , 4)), # (B, 32, 14, 17)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3 , 3)), # (B, 32, 12, 15)
            nn.MaxPool2d(kernel_size=(4 , 4)), # (B, 128, 3, 3)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=(3 , 3)), # (B, 32, 1, 1)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.lin_head= nn.Sequential(
            nn.Linear(128 + 256 + 256, 8),
        )
        self.softmax = nn.Softmax(dim = 1)
        self.transformer = Trans()

    def forward(self, x):
        cnn1out = self.cnn1(x)
        cnn2out = self.cnn2(x)
        transout = self.transformer(x)
        combined_features = torch.cat([cnn1out,cnn2out, transout], dim=1)
        logits = self.lin_head(combined_features)
        probs = self.softmax(logits)
        return logits, probs
