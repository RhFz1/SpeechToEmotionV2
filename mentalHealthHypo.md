# Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is an efficient algorithm to compute the Discrete Fourier Transform (DFT) and its inverse. The DFT is used to convert a sequence of complex numbers from the time domain to the frequency domain.

## Discrete Fourier Transform (DFT)

Given a sequence of \( N \) complex numbers \( x_0, x_1, x_2, \ldots, x_{N-1} \), the DFT is defined by:

\[
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2 \pi k n / N} \quad \text{for} \quad k = 0, 1, 2, \ldots, N-1
\]

where:
- \( X_k \) are the DFT coefficients.
- \( i \) is the imaginary unit with \( i^2 = -1 \).
- \( e^{-i 2 \pi k n / N} \) represents the complex exponential function.

## Inverse Discrete Fourier Transform (IDFT)

The inverse of the DFT, which converts frequency domain data back to the time domain, is given by:

\[
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i 2 \pi k n / N} \quad \text{for} \quad n = 0, 1, 2, \ldots, N-1
\]

## Fast Fourier Transform (FFT)

The FFT algorithm reduces the complexity of computing the DFT from \( O(N^2) \) to \( O(N \log N) \). This is achieved by recursively breaking down a DFT of any composite size \( N \) into many smaller DFTs.

### Cooley-Tukey Algorithm

The most common FFT algorithm is the Cooley-Tukey algorithm, which works by dividing the DFT into two smaller DFTs of even and odd indices. The steps are as follows:

1. **Divide**: Split the sequence \( x \) into two halves: even-indexed elements \( x_{\text{even}} \) and odd-indexed elements \( x_{\text{odd}} \).

\[
x_{\text{even}} = x_{0}, x_{2}, x_{4}, \ldots, x_{N-2}
\]
\[
x_{\text{odd}} = x_{1}, x_{3}, x_{5}, \ldots, x_{N-1}
\]

2. **Conquer**: Recursively compute the DFT of the two halves.
3. **Combine**: Combine the results to get the final DFT.

Mathematically, this is expressed as:

\[
X_k = X_k^{\text{even}} + e^{-i 2 \pi k / N} X_k^{\text{odd}}
\]
\[
X_{k+N/2} = X_k^{\text{even}} - e^{-i 2 \pi k / N} X_k^{\text{odd}}
\]

for \( k = 0, 1, 2, \ldots, N/2 - 1 \).

## Radix-2 FFT

The Radix-2 FFT is a special case of the Cooley-Tukey algorithm where the sequence length \( N \) is a power of 2. This makes the recursive division straightforward and highly efficient.

### Butterfly Diagram

The combination step in the Radix-2 FFT can be visualized using a butterfly diagram, which shows how pairs of values are combined and recombined at each stage.

### Example

For a sequence \( x \) of length 8 (i.e., \( N = 8 \)):

1. Divide into even and odd indexed elements.
2. Recursively apply FFT to each half.
3. Combine the results using the butterfly structure.

## Applications

FFT is widely used in various fields, including:
- Signal processing
- Image processing
- Solving partial differential equations
- Convolution operations

## Conclusion

The FFT is a powerful algorithm that significantly speeds up the computation of the DFT. Its recursive nature and efficient combination steps make it indispensable in modern computational applications.

For further reading and a deeper dive into FFT algorithms, refer to textbooks on numerical methods and signal processing.
