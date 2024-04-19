"""
Equation numbers refer to:

R. M. Mersereau, "The processing of hexagonally sampled two-dimensional signals," 
in Proceedings of the IEEE, vol. 67, no. 6, pp. 930-949, June 1979, doi: 10.1109/PROC.1979.11356.
"""

import numpy as np
from hexfft.utils import mersereau_region, hex_to_pgram, pgram_to_hex
from hexfft.array import HexArray

def fft(x, hexcrop=False, periodicity="rect", dtype=np.float32):

    if periodicity == "rect":
        assert hexcrop == False, "Cannot crop to hexagonal region when periodicity is rectangular."

    # use mersereau region
    if hexcrop:
        assert x.shape[0] == x.shape[1], "Must be a square region."
        N = x.shape[0]
        px = hex_to_pgram(x)
        PX = mersereau_fft(px)
        return pgram_to_hex(PX, N)

    # rectangular region
    assert x.shape[1] % 2 == 0
    return _rect_dft_slow(x)

def ifft(X, hexcrop=False, periodicity="rect", dtype=np.complex64):
    if periodicity == "rect":
        assert hexcrop == False, "Cannot crop to hexagonal region when periodicity is rectangular."

    # use mersereau region
    if hexcrop:
        assert X.shape[0] == X.shape[1], "Must be a square region."
        N = X.shape[0]
        PX = hex_to_pgram(X)
        px = mersereau_ifft(PX)
        return pgram_to_hex(px, N)

    # rectangular region
    assert X.shape[1] % 2 == 0
    return _rect_idft_slow(X)

def mersereau_fft(px):
    """"""
    assert 3 * px.shape[0] == px.shape[1], "must have dimensions Nx3N"
    M = px.shape[0]
    assert M % 2 == 0, "must be a multiple of 2"

    dtype = px.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    F = _hexdft_pgram(px[::2, ::2]).T
    G = _hexdft_pgram(px[::2, 1::2]).T
    H = _hexdft_pgram(px[1::2, ::2]).T
    I = _hexdft_pgram(px[1::2, 1::2]).T

    PX = HexArray(np.zeros_like(px.T, cdtype))

    # compute the sets of 4 indices which re-use the 
    # precomputed arrays above (eqns 49-52)
    L = np.zeros((3 * M, M), int)
    Q = int(3 * M/2)
    for i in range(int(M/2)):
        _ind = np.concatenate([np.arange(i * Q, (i+1)*Q)]*2)
        L[:, i] = _ind
        L[:, i + int(M//2)] = np.roll(_ind, M)

    for i in range(int(3*M**2 / 4)):
        # these 4 indices are 
        # k1, k2
        # k1 + 3M/2, k2
        # etc
        k1s, k2s = np.where(L == i)
        # find k1, k2 from eqns 49-52
        # this if-else block accounts for index wrapping
        if i % Q in np.arange(int(M//2)):
            k1, k2 = k1s[0], k2s[0]
            # sanity check
            assert k1s[1] == k1 + M
            assert k1s[2] == k1 + 3*M/2
            assert k1s[3] == k1 + 5*M/2

            FF = F[k1, k2]
            GG = W0(k1, k2, M)*G[k1, k2]
            HH = W1(k1, k2, M)*H[k1, k2]
            II = W2(k1, k2, M)*I[k1, k2]
            PX[k1s[0], k2s[0]] = FF + GG + HH + II
            PX[k1s[1], k2s[1]] = FF + GG - HH - II
            PX[k1s[2], k2s[2]] = FF - GG + HH - II
            PX[k1s[3], k2s[3]] = FF - GG - HH + II

        else:
            k1, k2 = k1s[1], k2s[1]
            # sanity check
            assert k1s[2] == (k1 + M)
            assert k1s[3] == (k1 + 3*M/2) 
            assert k1s[0] == (k1 + 5*M/2) % Q

            FF = F[k1, k2]
            GG = W0(k1, k2, M)*G[k1, k2]
            HH = W1(k1, k2, M)*H[k1, k2]
            II = W2(k1, k2, M)*I[k1, k2]
            PX[k1s[1], k2s[1]] = FF + GG + HH + II
            PX[k1s[2], k2s[2]] = FF + GG - HH - II
            PX[k1s[3], k2s[3]] = FF - GG + HH - II
            PX[k1s[0], k2s[0]] = FF - GG - HH + II

    return PX.T

def mersereau_ifft(PX):
    """"""
    assert 3*PX.shape[0] == PX.shape[1], "must have dimensions Nx3N"
    M = PX.shape[0]
    assert M % 2 == 0, "must be a multiple of 2"

    dtype = PX.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    F = _hexidft_pgram(PX[::2, ::2]).T
    G = _hexidft_pgram(PX[::2, 1::2]).T
    H = _hexidft_pgram(PX[1::2, ::2]).T
    I = _hexidft_pgram(PX[1::2, 1::2]).T

    px = HexArray(np.zeros_like(PX.T, cdtype))
    
    # compute the sets of 4 indices which re-use the 
    # precomputed arrays above (eqns 49-52)
    L = np.zeros((3 * M, M), int)
    Q = int(3 * M/2)
    for i in range(int(M/2)):
        _ind = np.concatenate([np.arange(i * Q, (i+1)*Q)]*2)
        L[:, i] = _ind
        L[:, i + int(M//2)] = np.roll(_ind, M)

    for i in range(int(3*M**2 / 4)):
        # these 4 indices are 
        # k1, k2
        # k1 + 3M/2, k2
        # etc
        k1s, k2s = np.where(L == i)
        # find k1, k2 from eqns 49-52
        # this if-else block accounts for index wrapping
        if i % Q in np.arange(int(M//2)):
            k1, k2 = k1s[0], k2s[0]
            # sanity check
            assert k1s[1] == k1 + M
            assert k1s[2] == k1 + 3*M/2
            assert k1s[3] == k1 + 5*M/2

            FF = F[k1, k2]
            GG = np.conj(W0(k1, k2, M))*G[k1, k2]
            HH = np.conj(W1(k1, k2, M))*H[k1, k2]
            II = np.conj(W2(k1, k2, M))*I[k1, k2]
            px[k1s[0], k2s[0]] = FF + GG + HH + II
            px[k1s[1], k2s[1]] = FF + GG - HH - II
            px[k1s[2], k2s[2]] = FF - GG + HH - II
            px[k1s[3], k2s[3]] = FF - GG - HH + II

        else:
            k1, k2 = k1s[1], k2s[1]
            # sanity check
            assert k1s[2] == (k1 + M)
            assert k1s[3] == (k1 + 3*M/2) 
            assert k1s[0] == (k1 + 5*M/2) % Q

            FF = F[k1, k2]
            GG = np.conj(W0(k1, k2, M))*G[k1, k2]
            HH = np.conj(W1(k1, k2, M))*H[k1, k2]
            II = np.conj(W2(k1, k2, M))*I[k1, k2]
            px[k1s[1], k2s[1]] = FF + GG + HH + II
            px[k1s[2], k2s[2]] = FF + GG - HH - II
            px[k1s[3], k2s[3]] = FF - GG + HH - II
            px[k1s[0], k2s[0]] = FF - GG - HH + II

    return px.T / 4
        
def fftshift(X):
    N = X.shape[0]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
    m = mersereau_region(X).astype(bool)

    shifted = HexArray(np.zeros_like(X))
    regI = (n1 < N//2) & (n2 < N//2)
    regII = m & (n1 < n2) & (n2 >= N // 2) 
    regIII = m & (n2 <= n1) & (n1 >= N//2)

    _regI = (n1 >= N//2) & (n2 >= N//2)
    _regII = m & (n1 >= n2) & (n2 < N//2)
    _regIII = m & (n2 > n1) & (n1 < N//2)

    shifted[_regI] = X[regI]
    shifted[_regII] = X[regII]
    shifted[_regIII] = X[regIII]

    return shifted

def ifftshift(X):
    N = X.shape[0]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
    m = mersereau_region(X).astype(bool)

    shifted = HexArray(np.zeros_like(X))
    _regI = (n1 < N//2) & (n2 < N//2)
    _regII = m & (n1 < n2) & (n2 >= N // 2) 
    _regIII = m & (n2 <= n1) & (n1 >= N//2)

    regI = (n1 >= N//2) & (n2 >= N//2)
    regII = m & (n1 >= n2) & (n2 < N//2)
    regIII = m & (n2 > n1) & (n1 < N//2)

    shifted[_regI] = X[regI]
    shifted[_regII] = X[regII]
    shifted[_regIII] = X[regIII]

    return shifted

# twiddle factors
def W0(k1, k2, M):
    return np.exp(-1.0j * 2 * np.pi * (2 * k2 - k1) / (3 * M))

def W1(k1, k2, M):
    return np.exp(-1.0j * 2 * np.pi * (2 * k1 - k2) / (3 * M))

def W2(k1, k2, M):
    return np.exp(-1.0j * 2 * np.pi * (k2 + k1) / (3 * M))

def hexdft(x):
    """"""
    return _hexdft_slow(x)

def hexidft(X):
    """"""
    return _hexidft_slow(X)

def _hexdft_slow(x):
    """
    Based on eqn (39) in Mersereau
    """
    assert x.shape[0] == x.shape[1], "must be square array"
    dtype = x.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    N = x.shape[0]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
    kern = _hexagonal_kernel(n1, n2, cdtype)

    support = mersereau_region(x)

    X = HexArray(np.zeros(x.shape, cdtype))
    for w1 in range(N):
        for w2 in range(N):
            X[w1, w2] = np.sum(kern[w1, w2, :, :] * x * support)

    return X

def _hexidft_slow(X):
    assert X.shape[0] == X.shape[1], "must be square array"
    cdtype = X.dtype
    dtype = np.float32 if cdtype == np.complex64 else np.float64

    N = X.shape[0]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))

    kern = np.conj(_hexagonal_kernel(n1, n2, cdtype))

    support = mersereau_region(X)

    x = HexArray(np.zeros(X.shape, cdtype))
    for x1 in range(N):
        for x2 in range(N):
            x[x1, x2] = np.sum(kern[:, :, x1, x2] * X * support)

    return x * (1 / np.sum(support)) * support

def _hexdft_pgram(px):
    """"""
    dtype = px.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    P = px.shape[0]
    p1, p2 = np.meshgrid(np.arange(3*P), np.arange(P))

    kern = _pgram_kernel(p1, p2, cdtype)

    X = np.zeros(px.shape, cdtype)
    for w1 in range(P):
        for w2 in range(3 * P):
            X[w1, w2] = np.sum(kern[w1, w2, :, :] * px)

    return X

def _hexidft_pgram(X):
    """"""
    cdtype = X.dtype
    dtype = np.float32 if cdtype == np.complex64 else np.float64

    P = X.shape[0]
    p1, p2 = np.meshgrid(np.arange(3*P), np.arange(P))
    kern = np.conj(_pgram_kernel(p1, p2, cdtype))

    px = np.zeros(X.shape, cdtype)
    for x1 in range(P):
        for x2 in range(3 * P):
            px[x1, x2] = np.sum(kern[:, :, x1, x2] * X)

    return px * 1/(3 * P**2)

def _hexagonal_kernel(n1, n2, cdtype):
    N = n1.shape[0]
    kernel = np.zeros((N,)*4, cdtype)
    # frequency indices
    w1s = np.arange(N)
    w2s = np.arange(N)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
            -1.0j * np.pi * ( 
            (1/(3*(N//2))) * (2*n1-n2)*(2*w1-w2) + (1/(N//2))*n2*w2
            )
        )
    return kernel

def _pgram_kernel(p1, p2, cdtype):
    _x, _y = p1.shape
    P = _x
    kernel = np.zeros(p1.shape + p1.shape, cdtype)
    w1s = np.arange(_x)
    w2s = np.arange(_y)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
            -1.0j * np.pi * ( 
            (1/(3*P)) * (2*p1-p2)*(2*w1-w2) + (1/P)*p2*w2
            )
        )
            
    return kernel

def rect_kernel(n1, n2, cdtype):
    N1, N2 = n1.shape
    kernel = np.zeros((N1, N2, N1, N2), cdtype)
    # frequency indices
    w1s = np.arange(N1)
    w2s = np.arange(N2)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
                -1.0j * 2* np.pi * ( 
                (1/(N1)) * w1*(n1 - n2/2) + (1/(N2))*(n2)*(w2)
            )
        )
    return kernel

def _rect_dft_slow(x):
    dtype = x.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    N1, N2 = x.shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    kern = rect_kernel(n1, n2, cdtype)
    X = np.zeros(x.shape, cdtype)
    for w1 in range(N1):
        for w2 in range(N2):
            X[w1, w2] = np.sum(kern[w1, w2, :, :] * x)

    return X

def _rect_idft_slow(X):
    cdtype = X.dtype
    
    N1, N2 = X.shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    kern = np.conj(rect_kernel(n1, n2, cdtype))
    x = np.zeros(X.shape, cdtype)
    for x1 in range(N1):
        for x2 in range(N2):
            x[x1, x2] = np.sum(kern[:, :, x1, x2] * X)

    return x * 1/(N1*N2)

