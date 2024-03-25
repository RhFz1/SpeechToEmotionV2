from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers.models.hubert.modeling_hubert import (
    HubertForSequenceClassification,
    HubertModel
)
import torch
import torch.nn as nn

model_name_or_path = "ntu-spml/distilhubert"
pooling_mode = "mean"


label_list = "neutral calm happy sad angry fearful disgust surprised".split()

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=8,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path,)
target_sampling_rate = feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()
        self.softmax = nn.Softmax(dim = 1)
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    def forward(self, x):
        out = self.hubert(x)
        x = torch.mean(out[0], dim=1)
        logits = self.classifier(x)
        probs = self.softmax(logits)
        
        return logits, probs

model = HubertForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

def load_model():
    return feature_extractor, model