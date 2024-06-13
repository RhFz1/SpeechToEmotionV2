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
```
