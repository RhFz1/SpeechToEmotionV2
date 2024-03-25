import torch
import os
import librosa
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv('SM_PATH')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Defining parameters needed for prediction.
model = torch.load(MODEL_PATH)
sampling_rate = 48000
subsample_duration = 3
emo_string = "neutral calm happy sad angry fearful disgust surprised"
idx_emo = emo_string.split()
emo_dict = {key: value for key, value in enumerate(idx_emo)}

def feature_melspectrogram(waveform, sample_rate = sampling_rate, mels=128):

    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_mels=mels, 
        fmax=sample_rate/2)
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def data_transform(file):

    # Loading the waveform from the file
    waveform, _ = librosa.load(file, offset=0.5 ,sr = sampling_rate)
    # Getting the duration of the file to split it.
    duration = librosa.get_duration(y=waveform)
    if duration < 1.000:
        return np.array([])
    # Finding number of splits that we can make.
    splits = int(duration) // subsample_duration

    segments = []

    for i in range(splits):
        homo_waveform = np.zeros((subsample_duration * sampling_rate,))
        temp_wave = waveform[sampling_rate * i * subsample_duration: sampling_rate * (i + 1) * subsample_duration]
        homo_waveform[:len(temp_wave)] = temp_wave
        segments.append(homo_waveform)

    if len(waveform) - (subsample_duration * splits * sampling_rate) > 0.33 * sampling_rate:
        homo_waveform = np.zeros((subsample_duration * sampling_rate))
        temp_wave = waveform[subsample_duration * splits * sampling_rate: ]
        homo_waveform[:len(temp_wave)] = temp_wave
        segments.append(homo_waveform) # (splits + 1, 3 * 48000)
    mels = []
    for segment in segments:
        spectrogram = feature_melspectrogram(segment)
        mels.append(spectrogram)
    # Shape mels (Splits + 1, 128, 282)
    batch = torch.tensor(np.array(mels), dtype=torch.float32).unsqueeze(1)
    return batch

def make_prediction(file):
    with torch.no_grad():
        X = data_transform(file)
        _, probs = model(X) # This return logits and probs, we need probs (B, Number of classes)
        preds = probs.mean(dim = 0) # this does mean along the batch dimension (Number of classes, )
        emotion = emo_dict[torch.argmax(preds).item()] # Getting the idx corresponding to the highest value and mapping it to the emotion.
        return emotion, preds