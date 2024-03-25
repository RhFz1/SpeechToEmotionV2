import torch
import librosa
import os, glob
import numpy as np
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler

load_dotenv()

DATA_PATH = os.environ.get('DATA_PATH')

sample_rate = 48000
emo_string = "neutral calm happy sad angry fearful disgust surprised"
idx_emo = emo_string.split()
emo_dict = {key: value for key, value in enumerate(idx_emo)}

data = ""

# Function responsible for generating mel spectrogram using waveforms
def feature_melspectrogram(waveform, sample_rate, mels=128):

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

def get_features(waveforms, features, samplerate):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mel = feature_melspectrogram(waveform, sample_rate)
        features.append(mel)
        file_count += 1
        # print progress 
        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    
    # return all features from list of waveforms
    return features

# Adding noise to the dataset to exclude the possibility of overfitting.
def awgn_augmentation(waveform, multiples=2, bits=16, snr_min=15, snr_max=30): 
    
    # get length of waveform (should be 3*48k = 144k)
    wave_len = len(waveform)
    
    # Generate normally distributed (Gaussian) noises
    # one for each waveform and multiple (i.e. wave_len*multiples noises)
    noise = np.random.normal(size=(multiples, wave_len))
    
    # Normalize waveform and noise
    norm_constant = 2.0**(bits-1)
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant
    
    # Compute power of waveform and power of noise
    signal_power = np.sum(norm_wave ** 2) / wave_len
    noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len
    
    # Choose random SNR in decibels in range [15,30]
    snr = np.random.randint(snr_min, snr_max)
    
    # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
    # Compute the covariance matrix used to whiten each noise 
    # actual SNR = signal/noise (power)
    # actual noise power = 10**(-snr/10)
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
    # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
    covariance = np.ones((wave_len, multiples)) * covariance

    # Since covariance and noise are arrays, * is the haddamard product 
    # Take Haddamard product of covariance and noise to generate white noise
    multiple_augmented_waveforms = waveform + covariance.T * noise
    
    return multiple_augmented_waveforms

def scaling(X_train, y_train, X_valid, y_valid, X_test, y_test):
    scaler = StandardScaler()
        #### Scale the training data ####
    # store shape so we can transform it back 
    N,C,H,W = X_train.shape
    # Reshape to 1D because StandardScaler operates on a 1D array
    # tell numpy to infer shape of 1D array with '-1' argument
    X_train = np.reshape(X_train, (N,-1)) 
    X_train = scaler.fit_transform(X_train)
    # Transform back to NxCxHxW 4D tensor format
    X_train = np.reshape(X_train, (N,C,H,W))

    ##### Scale the validation set ####
    N,C,H,W = X_valid.shape
    X_valid = np.reshape(X_valid, (N,-1))
    X_valid = scaler.transform(X_valid)
    X_valid = np.reshape(X_valid, (N,C,H,W))

    #### Scale the test set ####
    N,C,H,W = X_test.shape
    X_test = np.reshape(X_test, (N,-1))
    X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, (N,C,H,W))


    return X_train, y_train, X_valid, y_valid, X_test, y_test


def augment_waveforms(waveforms, features, emotions, multiples):
    # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
    emotion_count = 0
    # keep track of how many augmented samples we've added
    added_count = 0
    # convert emotion array to list for more efficient appending
    emotions = emotions.tolist()

    for waveform in waveforms:

        # Generate 2 augmented multiples of the dataset, i.e. 1440 native + 1440*2 noisy = 4320 samples total
        augmented_waveforms = awgn_augmentation(waveform, multiples=multiples)

        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:

            # Compute MFCCs over augmented waveforms
            augmented_mfcc = feature_melspectrogram(augmented_waveform, sample_rate=sample_rate)

            # append the augmented spectrogram to the rest of the native data
            features.append(augmented_mfcc)
            emotions.append(emotions[emotion_count])

            # keep track of new augmented samples
            added_count += 1

            # check progress
            print('\r'+f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms)*multiples} new augmented samples',end='')

        # keep track of the emotion labels to append in order
        emotion_count += 1

    return features, emotions
    
def load_data():
        
    filepath = "./loaded_datasets/melspectrogram.npy"
    
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
        waveform , _ = librosa.load(file, duration=3, offset=0.5, sr = sample_rate)
        waveform_homo = np.zeros((int(sample_rate * 3, )))
        waveform_homo[:len(waveform)] = waveform
        waveforms.append(waveform_homo), emotions.append(label)
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
        X_train.append(waveforms[train_indices,:])
        y_train.append(np.array([emotion_num]*len(train_indices),dtype=np.int32))
        # create validation waveforms/labels sets
        X_valid.append(waveforms[valid_indices,:])
        y_valid.append(np.array([emotion_num]*len(valid_indices),dtype=np.int32))
        # create test waveforms/labels sets
        X_test.append(waveforms[test_indices,:])
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

    # combine and store indices for all emotions' train, validation, test sets to verify uniqueness of sets
    train_set = np.concatenate(train_set,axis=0)
    valid_set = np.concatenate(valid_set,axis=0)
    test_set = np.concatenate(test_set,axis=0)

    # check shape of each set
    print(f'Training waveforms:{X_train.shape}, y_train:{y_train.shape}')
    print(f'Validation waveforms:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'Test waveforms:{X_test.shape}, y_test:{y_test.shape}')

    # make sure train, validation, test sets have no overlap/are unique
    # get all unique indices across all sets and how many times each index appears (count)
    uniques, count = np.unique(np.concatenate([train_set,test_set,valid_set],axis=0), return_counts=True)

    # if each index appears just once, and we have 1440 such unique indices, then all sets are unique
    if sum(count==1) == len(emotions):
        print(f'\nSets are unique: {sum(count==1)} samples out of {len(emotions)} are unique')
    else:
        print(f'\nSets are NOT unique: {sum(count==1)} samples out of {len(emotions)} are unique')
    

    # initialize feature arrays
    # We extract MEl features from waveforms and store in respective 'features' array
    features_train, features_valid, features_test = [],[],[]

    print('Train waveforms:') # get training set features 
    features_train = get_features(X_train, features_train, sample_rate)

    print('\n\nValidation waveforms:') # get validation set features
    features_valid = get_features(X_valid, features_valid, sample_rate)

    print('\n\nTest waveforms:') # get test set features 
    features_test = get_features(X_test, features_test, sample_rate)

    print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
    print(f'Features (Mel coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    # specify multiples of our dataset to add as augmented data
    multiples = 2

    print('Train waveforms:') # augment waveforms of training set
    features_train , y_train = augment_waveforms(X_train, features_train, y_train, multiples)

    print('\n\nValidation waveforms:') # augment waveforms of validation set
    features_valid, y_valid = augment_waveforms(X_valid, features_valid, y_valid, multiples)

    print('\n\nTest waveforms:') # augment waveforms of test set 
    features_test, y_test = augment_waveforms(X_test, features_test, y_test, multiples)

    # Check new shape of extracted features and data:
    print(f'\n\nNative + Augmented Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
    print(f'{len(y_train)} training sample labels, {len(y_valid)} validation sample labels, {len(y_test)} test sample labels')
    print(f'Features (MFCC matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    # need to make dummy input channel for CNN input feature tensor
    X_train = np.expand_dims(features_train,1)
    X_valid = np.expand_dims(features_valid, 1)
    X_test = np.expand_dims(features_test,1)

    # convert emotion labels from list back to numpy arrays for PyTorch to work with 
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # confiorm that we have tensor-ready 4D data array
    # should print (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
    print(f'Shape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
    print(f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

    X_train, y_train, X_valid, y_valid, X_test, y_test = scaling(X_train, y_train, X_valid, y_valid, X_test, y_test)

    del features_train, features_valid, features_test, waveforms

    with open(filepath, 'wb') as f:
        np.save(f, X_train)
        np.save(f, X_valid)
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, y_valid)
        np.save(f, y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


 

    