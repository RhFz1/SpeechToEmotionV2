import torch
import glob, os
import librosa
import numpy as np
from dotenv import load_dotenv
from hubert import feature_extractor as processor
from hubert import target_sampling_rate
load_dotenv()

DATA_PATH = os.environ.get('DATA_PATH')

duration = 5
emo_string = "neutral calm happy sad angry fearful disgust surprised"
idx_emo = emo_string.split()
emo_dict = {key: value for key, value in enumerate(idx_emo)}

def load_data():

    filepath = "loaded_datasets/waveforms.npy"

    if os.path.exists(filepath):
    # open file in read mode and read data 
        with open(filepath, 'rb') as f:
            X_train = np.load(f)
            X_valid = np.load(f)
            X_test = np.load(f)
            y_train = np.load(f)
            y_valid = np.load(f)
            y_test = np.load(f)
        return X_train, y_train, X_valid, y_valid, X_test, y_test   

    waveforms, emotions = [], []
    
    # Getting the raw waveforms and labels for each .wav file.
    cnt = 0
    for file in glob.glob(DATA_PATH):
        file_name = os.path.basename(file)
        label = int(file_name.split('-')[2]) - 1
        speech, sr = librosa.load(path=file, sr=target_sampling_rate)
        speech = processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                    return_tensors="pt", sampling_rate=sr).input_values
        speech = np.array(speech)
        waveforms.append(speech)
        emotions.append(label)
        cnt += 1
        print('\r' + f'Files processed {cnt}/1440', end = '')
        # Pushing these to X_train, y_train etc.
    X_train, y_train, X_valid, y_valid, X_test, y_test = [],[],[],[],[],[] 
    train_set, valid_set, test_set = [], [], []
    waveforms = np.array(waveforms)
    emotions = np.array(emotions , dtype=np.int32)
    
    # process each emotion separately to make sure we builf balanced train/valid/test sets 
    for emotion_num in range(len(idx_emo)):
            
        # find all indices of a single unique emotion
        emotion_indices = [index for index, emotion in enumerate(emotions) if emotion==emotion_num]

        # seed for reproducibility 
        np.random.seed(69)
        # shuffle indicies 
        emotion_indices = np.random.permutation(emotion_indices)

        # store dim (length) of the emotion list to make indices
        dim = len(emotion_indices)


        # store indices of training, validation and test sets in 80/10/10 proportion
        # train set is first 80%
        train_indices = emotion_indices[:int(0.8*dim)]
        # validation set is next 10% (between 80% and 90%)
        valid_indices = emotion_indices[int(0.8*dim):int(0.9*dim)]
        # test set is last 10% (between 90% - end/100%)
        test_indices = emotion_indices[int(0.9*dim):]

        train_indices = train_indices.astype(np.int32)
        valid_indices = valid_indices.astype(np.int32)
        test_indices = test_indices.astype(np.int32)

        # create train waveforms/labels sets
        X_train.append(waveforms[train_indices,:,:])
        y_train.append(np.array([emotion_num]*len(train_indices),dtype=np.int32))
        # create validation waveforms/labels sets
        X_valid.append(waveforms[valid_indices,:,:])
        y_valid.append(np.array([emotion_num]*len(valid_indices),dtype=np.int32))
        # create test waveforms/labels sets
        X_test.append(waveforms[test_indices,:,:])
        y_test.append(np.array([emotion_num]*len(test_indices),dtype=np.int32))

        # store indices for each emotion set to verify uniqueness between sets 
        train_set.append(train_indices)
        valid_set.append(valid_indices)
        test_set.append(test_indices)

    # concatenate, in order, all waveforms back into one array 
    X_train = np.concatenate(X_train,axis=0)
    X_valid = np.concatenate(X_valid,axis=0)
    X_test = np.concatenate(X_test,axis=0)

    # concatenate, in order, all emotions back into one array 
    y_train = np.concatenate(y_train,axis=0)
    y_valid = np.concatenate(y_valid,axis=0)
    y_test = np.concatenate(y_test,axis=0)

    # convert emotion labels from list back to numpy arrays for PyTorch to work with 
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    X_train = np.squeeze(X_train, axis = 1)
    X_valid = np.squeeze(X_valid, axis = 1)
    X_test = np.squeeze(X_test, axis = 1)

    with open(filepath, 'wb') as f:
        np.save(f, X_train)
        np.save(f, X_valid)
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, y_valid)
        np.save(f, y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test
