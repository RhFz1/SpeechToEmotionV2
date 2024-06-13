# Speech to Emotion Recognition

This project aims to recognize emotions from speech using deep learning techniques. The system converts audio samples into Mel Spectrograms and utilizes a deep learning model that combines Convolutional Neural Networks (CNN) and Transformer encoder blocks to classify emotions.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Mel Spectrograms](#mel-spectrograms)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset

The dataset used in this project comprises approximately 1450 labeled audio samples collected from open-source repositories such as RAVDESS (https://zenodo.org/records/1188976) and TESS (https://tspace.library.utoronto.ca/handle/1807/24487).

- **Audio Format**: .wav
- **Average Length**: 4 seconds per sample
- **Emotions**: 8 distinct emotions

## Preprocessing

Each audio sample in .wav format was converted into a Mel Spectrogram, through a series of operations such as FFT calculation and their concatenation. These were efficiently done thorugh a Python library known as Librosa.

## Mel Spectrograms

# Mel Spectrograms

A **Mel Spectrogram** is a representation of the short-term power spectrum of a sound signal, where the frequency axis is converted to the Mel scale. The Mel scale approximates the human ear's response to different frequencies, providing a more perceptually meaningful representation of sound.

## Steps to Compute a Mel Spectrogram

1. **Pre-emphasis**: This step is used to amplify the high frequencies. The pre-emphasis filter can be applied using the following equation:

    $$
    y(t) = x(t) - \alpha x(t-1)
    $$

    where \( x(t) \) is the input signal, \( y(t) \) is the output signal, and \( \alpha \) is the pre-emphasis coefficient (typically between 0.95 and 0.97).

2. **Framing**: The signal is divided into short frames of equal length. Each frame can be represented as:

    $$
    x[n] = x[n + i \cdot H], \quad i = 0, 1, 2, \ldots
    $$

    where \( H \) is the hop length (number of samples between successive frames), and \( n \) is the frame length.

3. **Windowing**: Each frame is multiplied by a window function (e.g., Hamming window) to reduce spectral leakage. The windowed signal \( w[n] \) is:

    $$
    w[n] = x[n] \cdot h[n]
    $$

    where \( h[n] \) is the window function.

4. **Fourier Transform and Power Spectrum**: The Discrete Fourier Transform (DFT) is applied to each windowed frame to obtain the frequency spectrum. The power spectrum \( P(f) \) is then computed:

    $$
    P(f) = |X(f)|^2
    $$

    where \( X(f) \) is the DFT of the windowed signal.

5. **Mel Filter Bank**: The power spectra are mapped onto the Mel scale using a filter bank. Each filter in the bank is triangular and corresponds to a point on the Mel scale. The Mel frequency \( m(f) \) is given by:

    $$
    m(f) = 2595 \log_{10} \left( 1 + \frac{f}{700} \right)
    $$

6. **Mel Spectrogram**: The result of applying the Mel filter bank to the power spectra gives the Mel spectrogram. The Mel spectrogram \( S_m \) can be computed as:

    $$
    S_m = M \cdot P
    $$

    where \( M \) is the Mel filter bank matrix and \( P \) is the power spectrum vector.

## Example Code (Python)

Here is a simple Python example using the `librosa` library to compute a Mel Spectrogram:

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
y, sr = librosa.load('audio_file.wav')

# Apply pre-emphasis filter
y_preemphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])

# Compute Short-Time Fourier Transform (STFT)
D = np.abs(librosa.stft(y_preemphasized))**2

# Compute Mel filter bank
mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)

# Compute Mel Spectrogram
S = np.dot(mel_basis, D)

# Convert to logarithmic scale (dB)
S_dB = librosa.power_to_db(S, ref=np.max)

# Display Mel Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()


### Fast Fourier Transform (FFT)

FFT is an algorithm that computes the Discrete Fourier Transform (DFT) and its inverse. FFT is used to convert the audio signal from its original domain (often time or space) to a representation in the frequency domain.

## Model Architecture

The deep learning model used for this project includes the following components:

1. **Two Convolutional Neural Networks (CNNs)**
2. **A Transformer Encoder Block**

[![App Platorm](https://github.com/IliaZenkov/transformer-cnn-emotion-recognition/blob/main/reports/cnn-transformer-final.png)]

Each of these networks operates independently and in parallel. Their results are concatenated to form a combined feature vector.

### CNNs



Convolutional Neural Networks (CNNs) are a type of deep learning model that are widely used for image and signal processing tasks. In this project, two CNNs are used to extract spatial features from the Mel Spectrogram images.
CNNs are particularly effective at capturing local patterns and features in images or spectrograms. They consist of convolutional layers, which apply learnable filters to the input data, and pooling layers, which downsample the feature maps. By stacking multiple convolutional and pooling layers, CNNs can learn increasingly abstract and complex representations of the input data.
In this model, the CNNs help in capturing the local patterns and spatial features present in the Mel Spectrograms, which are crucial for emotion recognition from speech.

To learn more about CNN's visit this page: https://en.wikipedia.org/wiki/Convolutional_neural_network

### Transformer Encoder Block

The Transformer Encoder is a type of neural network architecture originally introduced in the Transformer model for natural language processing tasks. In this project, a Transformer Encoder Block is used to capture the sequential dependencies and temporal context in the audio data.
The Transformer Encoder Block consists of several attention mechanisms, including multi-head self-attention and position-wise feed-forward networks. These attention mechanisms allow the model to weigh and combine different parts of the input sequence, enabling it to capture long-range dependencies and temporal relationships in the data.
By using a Transformer Encoder Block, the model can effectively model the temporal dynamics and context present in the speech signals, which is essential for accurate emotion recognition.

To learn more about Transformer visit this page: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

### Concatenation and Classification

The outputs of the CNNs and the Transformer Encoder Block are concatenated and passed through a series of dense layers. Finally, a softmax layer is applied to generate a probability distribution over the 8 emotion classes.

Softmax: The softmax function, also known as softargmax or normalized exponential function, converts a vector of K real numbers into a probability distribution of K possible outcomes. It is a generalization of the logistic function to multiple dimensions, and used in multinomial logistic regression. The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.

## Training

The model was trained using  roughly 4500 labelled augumented audio samples.
| Params/Model    | Epochs | lr    | l2_norm | Train Loss | Val Loss |
| -------  | ------- | -------- | --------| -------- | ------- |
| 8M  | 200    | 3e-4          |    0.01     |      93.62    |   68.65      |
| 16M| 150    |      3e-4    |      0.1   |    89.42      |     79.84    |
| 32M    | 100    |      3e-6    |  0.01       |     96.34     |     66.24    |

These are the stats we have achieved upon training various models with different architectures, there are more internal variations performed at model level but have not been highlighted in the table.



## Results

The performance of the model was evaluated on the validation set The model was able to predict the emotions with a reasonable accuracy of ~80%, demonstrating the effectiveness of combining CNNs and Transformer Encoders for this task.

This table demonstrates the results,

| Params/Model    | Val Loss |
| -------  | ------- |
| 8M  |  68.65      |
| 16M|79.84   |
| 32M    | 66.24    |

This clearly shows that a big model tends towards overfitting the data and requires more data to generalize, this we stick to a smaller model and tune it to achieve better results.


## Conclusion

This project demonstrates a novel approach to emotion recognition from speech by converting audio samples into Mel Spectrograms and using a hybrid deep learning model. The use of CNNs and Transformer Encoders allows the model to capture both spatial and temporal features, resulting in improved emotion classification.

