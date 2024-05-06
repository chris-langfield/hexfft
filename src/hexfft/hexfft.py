"""
Equation numbers refer to:

R. M. Mersereau, "The processing of hexagonally sampled two-dimensional signals," 
in Proceedings of the IEEE, vol. 67, no. 6, pp. 930-949, June 1979, doi: 10.1109/PROC.1979.11356.
"""

import numpy as np
import scipy
from hexfft.utils import hsupport, hex_to_pgram, pgram_to_hex, complex_type
from hexfft.array import HexArray, rect_shift, rect_unshift

import logging

_logger = logging.getLogger("hexfft")


def FFT(shape, periodicity="rect", dtype=np.complex128):
    if periodicity == "hex":
        return HexPeriodicFFT(shape, dtype)
    elif periodicity == "rect":
        return RectPeriodicFFT(shape, dtype)


class HexagonalFFT:
    def __init__(self, shape, dtype):
        assert len(shape) == 2, "Only 2D transforms are supported."
        assert dtype in [
            np.complex128,
            np.complex64,
        ], "dtype of transform must be complex"

        self.shape = shape
        self.dtype = dtype
        self.cdtype = complex_type(dtype)

        self.precompute()

    def precompute(self):
        self._precompute()

    def forward(self, x):
        return self._forward(x)

    def inverse(self, X):
        return self._inverse(X)


class HexPeriodicFFT(HexagonalFFT):
    def __init__(self, shape, dtype):
        assert (
            shape[0] == shape[1]
        ), "Input to FFT with hex periodicity must be a square array"
        assert (
            shape[0] % 4 == 0
        ), "Array size for periodicity='hex' must be a multiple of 4."

        super().__init__(shape, dtype)

    def _precompute(self):
        N = self.shape[0]
        M = N // 2

        self.L = np.zeros((3 * M, M), int)
        Q = int(3 * M / 2)

        for i in range(int(M / 2)):
            _ind = np.concatenate([np.arange(i * Q, (i + 1) * Q)] * 2)
            self.L[:, i] = _ind
            self.L[:, i + int(M // 2)] = np.roll(_ind, M)

        k1, k2 = np.indices(self.L.shape)

        self.W0 = np.exp(-1.0j * 2 * np.pi * (2 * k2 - k1) / (3 * M)).astype(
            self.cdtype
        )
        self.W1 = np.exp(-1.0j * 2 * np.pi * (2 * k1 - k2) / (3 * M)).astype(
            self.cdtype
        )
        self.W2 = np.exp(-1.0j * 2 * np.pi * (k2 + k1) / (3 * M)).astype(self.cdtype)

        self.conj_W0 = np.conj(self.W0)
        self.conj_W1 = np.conj(self.W1)
        self.conj_W2 = np.conj(self.W2)

    def _forward(self, x):
        assert (
            x.shape[-2:] == self.shape
        ), f"Input array with shape {x.shape} does not match FFT object shape {self.shape}"

        N = self.shape[0]

        squeeze = x.ndim == 2
        if squeeze:
            x = np.expand_dims(x, 0)

        px = hex_to_pgram(x)

        assert 3 * px.shape[1] == px.shape[2], "must have dimensions Nx3N"
        M = px.shape[1]
        Q = int(3 * M / 2)
        assert M % 2 == 0, "must be a multiple of 2"
        assert M == N // 2

        F = np.transpose(_hexdft_pgram_stack(px[:, ::2, ::2]), axes=(0, 2, 1))
        G = np.transpose(_hexdft_pgram_stack(px[:, ::2, 1::2]), axes=(0, 2, 1))
        H = np.transpose(_hexdft_pgram_stack(px[:, 1::2, ::2]), axes=(0, 2, 1))
        I = np.transpose(_hexdft_pgram_stack(px[:, 1::2, 1::2]), axes=(0, 2, 1))

        PX = HexArray(
            np.zeros_like(np.transpose(px, axes=(0, 2, 1)), self.dtype),
            pattern=x.pattern,
        )

        for i in range(int(3 * M**2 / 4)):
            k1s, k2s = np.where(self.L == i)
            if i % Q in np.arange(int(M // 2)):
                k1, k2 = k1s[0], k2s[0]
                # sanity check
                assert k1s[1] == k1 + M
                assert k1s[2] == k1 + 3 * M / 2
                assert k1s[3] == k1 + 5 * M / 2

                FF = F[:, k1, k2]
                GG = self.W0[k1, k2] * G[:, k1, k2]
                HH = self.W1[k1, k2] * H[:, k1, k2]
                II = self.W2[k1, k2] * I[:, k1, k2]
                PX[:, k1s[0], k2s[0]] = FF + GG + HH + II
                PX[:, k1s[1], k2s[1]] = FF + GG - HH - II
                PX[:, k1s[2], k2s[2]] = FF - GG + HH - II
                PX[:, k1s[3], k2s[3]] = FF - GG - HH + II

            else:
                k1, k2 = k1s[1], k2s[1]
                # sanity check
                assert k1s[2] == (k1 + M)
                assert k1s[3] == (k1 + 3 * M / 2)
                assert k1s[0] == (k1 + 5 * M / 2) % Q

                FF = F[:, k1, k2]
                GG = self.W0[k1, k2] * G[:, k1, k2]
                HH = self.W1[k1, k2] * H[:, k1, k2]
                II = self.W2[k1, k2] * I[:, k1, k2]
                PX[:, k1s[1], k2s[1]] = FF + GG + HH + II
                PX[:, k1s[2], k2s[2]] = FF + GG - HH - II
                PX[:, k1s[3], k2s[3]] = FF - GG + HH - II
                PX[:, k1s[0], k2s[0]] = FF - GG - HH + II

        X = pgram_to_hex(np.transpose(PX, (0, 2, 1)), N, x.pattern)

        if squeeze:
            X = np.squeeze(X)

        return X

    def _inverse(self, X):
        assert (
            X.shape[-1] == X.shape[-2]
        ), "Input to FFT with hex periodicity must be a square array"
        N = X.shape[-1]
        assert N % 4 == 0, "Array size for periodicity='hex' must be a multiple of 4."

        assert (
            X.shape[-2:] == self.shape
        ), f"Input array with shape {X.shape} does not match FFT object shape {self.shape}"

        squeeze = X.ndim == 2
        if squeeze:
            X = np.expand_dims(X, 0)

        PX = hex_to_pgram(X)
        assert 3 * PX.shape[1] == PX.shape[2], "must have dimensions Nx3N"
        M = PX.shape[1]
        Q = int(3 * M / 2)
        assert M % 2 == 0, "must be a multiple of 2"
        assert M == N // 2

        F = np.transpose(_hexidft_pgram_stack(PX[:, ::2, ::2]), axes=(0, 2, 1))
        G = np.transpose(_hexidft_pgram_stack(PX[:, ::2, 1::2]), axes=(0, 2, 1))
        H = np.transpose(_hexidft_pgram_stack(PX[:, 1::2, ::2]), axes=(0, 2, 1))
        I = np.transpose(_hexidft_pgram_stack(PX[:, 1::2, 1::2]), axes=(0, 2, 1))

        px = HexArray(
            np.zeros_like(np.transpose(PX, axes=(0, 2, 1)), self.dtype),
            pattern="oblique",
        )

        for i in range(int(3 * M**2 / 4)):
            k1s, k2s = np.where(self.L == i)
            if i % Q in np.arange(int(M // 2)):
                k1, k2 = k1s[0], k2s[0]
                # sanity check
                assert k1s[1] == k1 + M
                assert k1s[2] == k1 + 3 * M / 2
                assert k1s[3] == k1 + 5 * M / 2

                FF = F[:, k1, k2]
                GG = self.conj_W0[k1, k2] * G[:, k1, k2]
                HH = self.conj_W1[k1, k2] * H[:, k1, k2]
                II = self.conj_W2[k1, k2] * I[:, k1, k2]
                px[:, k1s[0], k2s[0]] = FF + GG + HH + II
                px[:, k1s[1], k2s[1]] = FF + GG - HH - II
                px[:, k1s[2], k2s[2]] = FF - GG + HH - II
                px[:, k1s[3], k2s[3]] = FF - GG - HH + II

            else:
                k1, k2 = k1s[1], k2s[1]
                # sanity check
                assert k1s[2] == (k1 + M)
                assert k1s[3] == (k1 + 3 * M / 2)
                assert k1s[0] == (k1 + 5 * M / 2) % Q

                FF = F[:, k1, k2]
                GG = self.conj_W0[k1, k2] * G[:, k1, k2]
                HH = self.conj_W1[k1, k2] * H[:, k1, k2]
                II = self.conj_W2[k1, k2] * I[:, k1, k2]
                px[:, k1s[1], k2s[1]] = FF + GG + HH + II
                px[:, k1s[2], k2s[2]] = FF + GG - HH - II
                px[:, k1s[3], k2s[3]] = FF - GG + HH - II
                px[:, k1s[0], k2s[0]] = FF - GG - HH + II

        x = pgram_to_hex(np.transpose(px, (0, 2, 1)), N, X.pattern) / 4.0

        if squeeze:
            x = np.squeeze(x)

        return x


class RectPeriodicFFT(HexagonalFFT):
    def __init__(self, shape, dtype):
        super().__init__(shape, dtype)

    def _precompute(self):
        N1, N2 = self.shape
        self.phase_shift = np.exp(
            1.0j * np.pi * np.array([i * np.arange(N2) for i in range(N1)]) / N2
        )
        self.phase_shift_conj = np.conj(self.phase_shift)

    def _forward(self, x):
        assert (
            x.shape[-2:] == self.shape
        ), f"Input array with shape {x.shape} does not match FFT object shape {self.shape}"
        assert isinstance(
            x, HexArray
        ), "Input to rectangular periodic FFT must be HexArray."

        shift = x.pattern == "offset"
        if shift:
            x = rect_shift(x)

        squeeze = x.ndim == 2
        if squeeze:
            x = np.expand_dims(x, 0)

        F1 = scipy.fft.fft(x, axis=2)
        F2 = F1 * self.phase_shift
        X = scipy.fft.fft(F2, axis=1)

        if squeeze:
            X = np.squeeze(X)

        if shift:
            X = rect_unshift(HexArray(X, "oblique"))
        else:
            X = HexArray(X, "oblique")

        return X

    def _inverse(self, X):
        assert (
            X.shape[-2:] == self.shape
        ), f"Input array with shape {X.shape} does not match FFT object shape {self.shape}"
        assert isinstance(
            X, HexArray
        ), "Input to rectangular periodic FFT must be HexArray."

        shift = X.pattern == "offset"
        if shift:
            X = rect_shift(X)

        squeeze = X.ndim == 2
        if squeeze:
            X = np.expand_dims(X, 0)

        F2 = scipy.fft.ifft(X, axis=1)
        F1 = F2 * self.phase_shift_conj
        x = HexArray(scipy.fft.ifft(F1, axis=2), "oblique")

        if squeeze:
            x = np.squeeze(x)

        if shift:
            x = rect_unshift(HexArray(x, "oblique"))
        else:
            x = HexArray(x, "oblique")

        return x


def fft(x, periodicity="rect"):
    if periodicity == "hex":
        assert x.shape[0] == x.shape[1], "Input to fft(-, 'hex') must be a square array"
        N = x.shape[0]
        assert N % 4 == 0, "Array size for periodicity='hex' must be a multiple of 4."

        px = hex_to_pgram(x)
        PX = mersereau_fft(px)
        return pgram_to_hex(PX, N, pattern=x.pattern)

    elif periodicity == "rect":
        assert isinstance(
            x, HexArray
        ), "Input to rectangular periodic FFT must be HexArray."

        shift = x.pattern == "offset"
        if shift:
            x = rect_shift(x)

        X = rect_fft(x)
        if shift:
            X = rect_unshift(X)

        return X

    raise ValueError(f"Unrecognized periodicity option: {periodicity}")


def ifft(X, periodicity="rect"):
    if periodicity == "hex":
        assert (
            X.shape[0] == X.shape[1]
        ), "Input to ifft(-, 'hex') must be a square array"
        N = X.shape[0]
        assert N % 4 == 0, "Array size for periodicity='hex' must be a multiple of 4."

        PX = hex_to_pgram(X)
        px = mersereau_ifft(PX)
        return pgram_to_hex(px, N, pattern=X.pattern)

    elif periodicity == "rect":
        assert isinstance(
            X, HexArray
        ), "Input to rectangular periodic IFFT must be HexArray."
        shift = X.pattern == "offset"
        if shift:
            X = rect_shift(X)

        x = rect_ifft(X)

        if shift:
            x = rect_unshift(x)

        return x

    raise ValueError(f"Unrecognized periodicity option: {periodicity}")


def mersereau_fft(px):
    """"""
    assert 3 * px.shape[0] == px.shape[1], "must have dimensions Nx3N"
    M = px.shape[0]
    assert M % 2 == 0, "must be a multiple of 2"

    dtype = px.dtype
    cdtype = complex_type(dtype)

    F = _hexdft_pgram(px[::2, ::2]).T
    G = _hexdft_pgram(px[::2, 1::2]).T
    H = _hexdft_pgram(px[1::2, ::2]).T
    I = _hexdft_pgram(px[1::2, 1::2]).T

    PX = HexArray(np.zeros_like(px.T, cdtype), pattern="oblique")

    # compute the sets of 4 indices which re-use the
    # precomputed arrays above (eqns 49-52)
    L = np.zeros((3 * M, M), int)
    Q = int(3 * M / 2)
    for i in range(int(M / 2)):
        _ind = np.concatenate([np.arange(i * Q, (i + 1) * Q)] * 2)
        L[:, i] = _ind
        L[:, i + int(M // 2)] = np.roll(_ind, M)

    k1, k2 = np.indices(L.shape)
    W0 = np.exp(-1.0j * 2 * np.pi * (2 * k2 - k1) / (3 * M)).astype(cdtype)
    W1 = np.exp(-1.0j * 2 * np.pi * (2 * k1 - k2) / (3 * M)).astype(cdtype)
    W2 = np.exp(-1.0j * 2 * np.pi * (k2 + k1) / (3 * M)).astype(cdtype)

    for i in range(int(3 * M**2 / 4)):
        # these 4 indices are
        # k1, k2
        # k1 + 3M/2, k2
        # etc
        k1s, k2s = np.where(L == i)
        # find k1, k2 from eqns 49-52
        # this if-else block accounts for index wrapping
        if i % Q in np.arange(int(M // 2)):
            k1, k2 = k1s[0], k2s[0]
            # sanity check
            assert k1s[1] == k1 + M
            assert k1s[2] == k1 + 3 * M / 2
            assert k1s[3] == k1 + 5 * M / 2

            FF = F[k1, k2]
            GG = W0[k1, k2] * G[k1, k2]
            HH = W1[k1, k2] * H[k1, k2]
            II = W2[k1, k2] * I[k1, k2]
            PX[k1s[0], k2s[0]] = FF + GG + HH + II
            PX[k1s[1], k2s[1]] = FF + GG - HH - II
            PX[k1s[2], k2s[2]] = FF - GG + HH - II
            PX[k1s[3], k2s[3]] = FF - GG - HH + II

        else:
            k1, k2 = k1s[1], k2s[1]
            # sanity check
            assert k1s[2] == (k1 + M)
            assert k1s[3] == (k1 + 3 * M / 2)
            assert k1s[0] == (k1 + 5 * M / 2) % Q

            FF = F[k1, k2]
            GG = W0[k1, k2] * G[k1, k2]
            HH = W1[k1, k2] * H[k1, k2]
            II = W2[k1, k2] * I[k1, k2]
            PX[k1s[1], k2s[1]] = FF + GG + HH + II
            PX[k1s[2], k2s[2]] = FF + GG - HH - II
            PX[k1s[3], k2s[3]] = FF - GG + HH - II
            PX[k1s[0], k2s[0]] = FF - GG - HH + II

    return PX.T


def mersereau_ifft(PX):
    """"""
    assert 3 * PX.shape[0] == PX.shape[1], "must have dimensions Nx3N"
    M = PX.shape[0]
    assert M % 2 == 0, "must be a multiple of 2"

    dtype = PX.dtype
    cdtype = complex_type(dtype)

    F = _hexidft_pgram(PX[::2, ::2]).T
    G = _hexidft_pgram(PX[::2, 1::2]).T
    H = _hexidft_pgram(PX[1::2, ::2]).T
    I = _hexidft_pgram(PX[1::2, 1::2]).T

    px = HexArray(np.zeros_like(PX.T, cdtype), pattern="oblique")

    # compute the sets of 4 indices which re-use the
    # precomputed arrays above (eqns 49-52)
    L = np.zeros((3 * M, M), int)
    Q = int(3 * M / 2)
    for i in range(int(M / 2)):
        _ind = np.concatenate([np.arange(i * Q, (i + 1) * Q)] * 2)
        L[:, i] = _ind
        L[:, i + int(M // 2)] = np.roll(_ind, M)

    k1, k2 = np.indices(L.shape)
    W0 = np.exp(-1.0j * 2 * np.pi * (2 * k2 - k1) / (3 * M)).astype(cdtype)
    W1 = np.exp(-1.0j * 2 * np.pi * (2 * k1 - k2) / (3 * M)).astype(cdtype)
    W2 = np.exp(-1.0j * 2 * np.pi * (k2 + k1) / (3 * M)).astype(cdtype)

    for i in range(int(3 * M**2 / 4)):
        # these 4 indices are
        # k1, k2
        # k1 + 3M/2, k2
        # etc
        k1s, k2s = np.where(L == i)
        # find k1, k2 from eqns 49-52
        # this if-else block accounts for index wrapping
        if i % Q in np.arange(int(M // 2)):
            k1, k2 = k1s[0], k2s[0]
            # sanity check
            assert k1s[1] == k1 + M
            assert k1s[2] == k1 + 3 * M / 2
            assert k1s[3] == k1 + 5 * M / 2

            FF = F[k1, k2]
            GG = np.conj(W0[k1, k2]) * G[k1, k2]
            HH = np.conj(W1[k1, k2]) * H[k1, k2]
            II = np.conj(W2[k1, k2]) * I[k1, k2]
            px[k1s[0], k2s[0]] = FF + GG + HH + II
            px[k1s[1], k2s[1]] = FF + GG - HH - II
            px[k1s[2], k2s[2]] = FF - GG + HH - II
            px[k1s[3], k2s[3]] = FF - GG - HH + II

        else:
            k1, k2 = k1s[1], k2s[1]
            # sanity check
            assert k1s[2] == (k1 + M)
            assert k1s[3] == (k1 + 3 * M / 2)
            assert k1s[0] == (k1 + 5 * M / 2) % Q

            FF = F[k1, k2]
            GG = np.conj(W0[k1, k2]) * G[k1, k2]
            HH = np.conj(W1[k1, k2]) * H[k1, k2]
            II = np.conj(W2[k1, k2]) * I[k1, k2]
            px[k1s[1], k2s[1]] = FF + GG + HH + II
            px[k1s[2], k2s[2]] = FF + GG - HH - II
            px[k1s[3], k2s[3]] = FF - GG + HH - II
            px[k1s[0], k2s[0]] = FF - GG - HH + II

    return px.T / 4


def fftshift(X):
    N = X.shape[0]
    n1, n2 = X.indices
    m = hsupport(N, X.pattern).astype(bool)
    shifted = HexArray(np.zeros_like(X), X.pattern)
    if X.pattern == "oblique":
        regI = (n1 < N // 2) & (n2 < N // 2)
        regII = m & (n1 < n2) & (n2 >= N // 2)
        regIII = m & (n2 <= n1) & (n1 >= N // 2)

        _regI = (n1 >= N // 2) & (n2 >= N // 2)
        _regII = m & (n1 >= n2) & (n2 < N // 2)
        _regIII = m & (n2 > n1) & (n1 < N // 2)

        shifted[_regI] = X[regI]
        shifted[_regII] = X[regII]
        shifted[_regIII] = X[regIII]

    elif X.pattern == "offset":
        m = m.T
        n2 = n2 - N // 4
        regI = m & (n1 < N // 2) & (n2 < N // 2)
        regII = m & (n1 <= n2) & (n2 >= N // 2)
        regIII = m & (n2 < n1) & (n1 >= N // 2)

        _regI = m & (n1 >= N // 2) & (n2 >= N // 2)
        _regII = m & (n1 > n2) & (n2 < N // 2)
        _regIII = m & (n2 >= n1) & (n1 < N // 2)

        shifted[_regI.T] = X[regI.T]
        shifted[_regII.T] = X[regII.T]
        shifted[_regIII.T] = X[regIII.T]

    return shifted


def ifftshift(X):
    N = X.shape[0]
    n1, n2 = X.indices
    m = hsupport(N, X.pattern).astype(bool)
    shifted = HexArray(np.zeros_like(X), X.pattern)

    if X.pattern == "oblique":
        _regI = (n1 < N // 2) & (n2 < N // 2)
        _regII = m & (n1 < n2) & (n2 >= N // 2)
        _regIII = m & (n2 <= n1) & (n1 >= N // 2)

        regI = (n1 >= N // 2) & (n2 >= N // 2)
        regII = m & (n1 >= n2) & (n2 < N // 2)
        regIII = m & (n2 > n1) & (n1 < N // 2)

        shifted[_regI] = X[regI]
        shifted[_regII] = X[regII]
        shifted[_regIII] = X[regIII]

    elif X.pattern == "offset":
        m = m.T
        n2 = n2 - N // 4
        regI = m & (n1 < N // 2) & (n2 < N // 2)
        regII = m & (n1 <= n2) & (n2 >= N // 2)
        regIII = m & (n2 < n1) & (n1 >= N // 2)

        _regI = m & (n1 >= N // 2) & (n2 >= N // 2)
        _regII = m & (n1 > n2) & (n2 < N // 2)
        _regIII = m & (n2 >= n1) & (n1 < N // 2)

        shifted[regI.T] = X[_regI.T]
        shifted[regII.T] = X[_regII.T]
        shifted[regIII.T] = X[_regIII.T]

    return shifted


def _hexdft_pgram(px):
    """"""
    dtype = px.dtype
    cdtype = complex_type(dtype)

    P = px.shape[0]
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))

    kern = _pgram_kernel(p1, p2, cdtype)

    X = np.zeros(px.shape, cdtype)
    for w1 in range(P):
        for w2 in range(3 * P):
            X[w1, w2] = np.sum(kern[w1, w2, :, :] * px)

    return X


def _hexdft_pgram_stack(px):
    """"""
    dtype = px.dtype
    cdtype = complex_type(dtype)

    nstack, P, _ = px.shape
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))

    kern = _pgram_kernel(p1, p2, cdtype)

    X = np.zeros(px.shape, cdtype)
    for w1 in range(P):
        for w2 in range(3 * P):
            X[:, w1, w2] = np.sum(kern[w1, w2, :, :][None, :, :] * px, axis=(1, 2))

    return X


def _hexidft_pgram(X):
    """"""
    dtype = X.dtype
    cdtype = complex_type(dtype)

    P = X.shape[0]
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))
    kern = np.conj(_pgram_kernel(p1, p2, cdtype))

    px = np.zeros(X.shape, cdtype)
    for x1 in range(P):
        for x2 in range(3 * P):
            px[x1, x2] = np.sum(kern[:, :, x1, x2] * X)

    return px * 1 / (3 * P**2)


def _hexidft_pgram_stack(X):
    """"""
    dtype = X.dtype
    cdtype = complex_type(dtype)

    nstack, P, _ = X.shape
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))
    kern = np.conj(_pgram_kernel(p1, p2, cdtype))

    px = np.zeros(X.shape, cdtype)
    for x1 in range(P):
        for x2 in range(3 * P):
            px[:, x1, x2] = np.sum(kern[:, :, x1, x2][None, :, :] * X, axis=(1, 2))

    return px * 1 / (3 * P**2)


def _pgram_kernel(p1, p2, cdtype):
    _x, _y = p1.shape
    P = _x
    kernel = np.zeros(p1.shape + p1.shape, cdtype)
    w1s = np.arange(_x)
    w2s = np.arange(_y)
    for w1 in w1s:
        for w2 in w2s:
            kernel[w1, w2, :, :] = np.exp(
                -1.0j
                * np.pi
                * ((1 / (3 * P)) * (2 * p1 - p2) * (2 * w1 - w2) + (1 / P) * p2 * w2)
            )

    return kernel


def rect_fft(x):
    dtype = x.dtype
    cdtype = complex_type(dtype)

    N1, N2 = x.shape
    F1 = scipy.fft.fft(x, axis=1)
    exp_factor = np.exp(
        1.0j * np.pi * np.array([i * np.arange(N2) for i in range(N1)]) / N2
    ).astype(cdtype)
    F2 = F1 * exp_factor
    F = HexArray(scipy.fft.fft(F2, axis=0), "oblique")

    return F


def rect_ifft(X):
    dtype = X.dtype
    cdtype = complex_type(dtype)

    N1, N2 = X.shape
    F2 = scipy.fft.ifft(X, axis=0)
    exp_factor = np.exp(
        -1.0j * np.pi * np.array([i * np.arange(N2) for i in range(N1)]) / N2
    ).astype(cdtype)
    F1 = F2 * exp_factor
    x = HexArray(scipy.fft.ifft(F1, axis=1), "oblique")

    return x
