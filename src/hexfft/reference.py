import numpy as np
from hexfft import HexArray
from hexfft.utils import hsupport


def _rect_kernel(n1, n2, cdtype):
    N1, N2 = n1.shape
    kernel = np.zeros((N1, N2, N1, N2), cdtype)
    # frequency indices
    w1s = np.arange(N1)
    w2s = np.arange(N2)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
                -1.0j
                * 2
                * np.pi
                * ((1 / (N1)) * w1 * (n1 - n2 / 2) + (1 / (N2)) * (n2) * (w2))
            )
    return kernel


def _hexagonal_kernel(n1, n2, cdtype):
    N = n1.shape[0]
    kernel = np.zeros((N,) * 4, cdtype)
    # frequency indices
    w1s = np.arange(N)
    w2s = np.arange(N)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
                -1.0j
                * np.pi
                * (
                    (1 / (3 * (N // 2))) * (2 * n1 - n2) * (2 * w1 - w2)
                    + (1 / (N // 2)) * n2 * w2
                )
            )
    return kernel


def _rect_dft_slow(x):
    dtype = x.dtype
    cdtype = np.complex64 if dtype == np.float32 else np.complex128

    N1, N2 = x.shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    kern = _rect_kernel(n1, n2, cdtype)
    X = HexArray(np.zeros(x.shape, cdtype), "oblique")
    for w1 in range(N1):
        for w2 in range(N2):
            X[w1, w2] = np.sum(kern[w1, w2, :, :] * x)

    return X


def _rect_idft_slow(X):
    cdtype = X.dtype

    N1, N2 = X.shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    kern = np.conj(_rect_kernel(n1, n2, cdtype))
    x = HexArray(np.zeros(X.shape, cdtype), "oblique")
    for x1 in range(N1):
        for x2 in range(N2):
            x[x1, x2] = np.sum(kern[:, :, x1, x2] * X)

    return x * 1 / (N1 * N2)


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

    support = hsupport(N)

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

    support = hsupport(N)

    x = HexArray(np.zeros(X.shape, cdtype))
    for x1 in range(N):
        for x2 in range(N):
            x[x1, x2] = np.sum(kern[:, :, x1, x2] * X * support)

    return x * (1 / np.sum(support)) * support
