from hexfft import fftshift, ifftshift, HexArray
from hexfft.hexfft import (
    _hexdft_pgram,
    _hexidft_pgram,
    _rect_dft_slow,
    _rect_idft_slow,
    mersereau_fft,
    mersereau_ifft,
    hexdft,
    hexidft,
)
from hexfft.utils import (
    mersereau_region,
    pgram_to_hex,
    nice_test_function,
    hex_to_pgram,
)
from hexfft.array import rect_shift, rect_unshift
import numpy as np


def hregion(n1, n2, center, size):
    """
    return mask for a hexagonal region of support
    with side length size centered at center
    """
    h1, h2 = center
    A = n2 < h2 + size
    B = n2 > h2 - size
    C = n1 > h1 - size
    D = n1 < h2 + size
    E = n2 < n1 + (h2 - h1) + size
    F = n2 > n1 + (h2 - h1) - size
    cond = A & B & C & D & E & F
    return HexArray(cond.astype(int))


def test_slow_hexdft():
    # testing in float64

    for size in [5, 6, 20, 21, 50, 51]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        N = n1.shape[0]
        if N % 2 == 1:
            center = (N // 2, N // 2)
        else:
            center = (N / 2 - 1, N / 2 - 1)

        impulse = hregion(n1, n2, center, 1)

        IMPULSE = hexdft(impulse)
        impulse_T = hexidft(IMPULSE)

        m = mersereau_region(impulse)

        assert np.allclose(impulse * m, impulse_T * m, atol=1e-12)


def test_pgram_hexdft():
    # test dft on 3N x N parallelogram

    for size in [5, 6, 20, 21, 50, 51]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        N = n1.shape[0]
        if N % 2 == 1:
            center = (N // 2, N // 2)
        else:
            center = (N / 2 - 1, N / 2 - 1)
        impulse = hregion(n1, n2, center, 1)

        impulse_p = hex_to_pgram(impulse)

        IMPULSE_P = _hexdft_pgram(impulse_p)
        impulse_p_T = _hexidft_pgram(IMPULSE_P)

        assert np.allclose(impulse_p, impulse_p_T, atol=1e-12)


def test_rect_hexdft():
    # test rect dft and idft
    for size in [4, 5, 8, 9, 16, 17]:
        N1 = size
        # N2 must be even
        N2 = N1 + N1 % 2
        n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
        center = (N1 / 2 - 1, N2 / 2 - 1)
        d = hregion(n1, n2, center, 1)

        D = _rect_dft_slow(d)
        dd = _rect_idft_slow(D)

        assert np.allclose(d, dd, atol=1e-12)


def test_mersereau_fft():
    # testing in float64

    for size in [4, 8, 16, 32]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        N = n1.shape[0]
        center = (N / 2 - 1, N / 2 - 1)

        # test on an impulse function and on an arbitrary function
        # d for "dirac"
        d = hregion(n1, n2, center, 1)
        x = nice_test_function(n1, n2)
        pd = hex_to_pgram(d)
        px = hex_to_pgram(x)

        # compare forward transformation
        D_SLOW = _hexdft_pgram(pd)
        D = mersereau_fft(pd)
        X_SLOW = _hexdft_pgram(px)
        X = mersereau_fft(px)
        assert np.allclose(D_SLOW, D, atol=1e-12)
        assert np.allclose(X_SLOW, X, atol=1e-12)

        # compare inverse transform
        dd_slow = _hexidft_pgram(D)
        dd = mersereau_ifft(D)
        xx_slow = _hexidft_pgram(X)
        xx = mersereau_ifft(X)
        assert np.allclose(dd_slow, dd, atol=1e-12)
        assert np.allclose(xx_slow, xx, atol=1e-12)


def test_fftshift():
    for size in [8, 16, 32]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        x = nice_test_function(n1, n2)
        h_oblique = HexArray(x) * mersereau_region(HexArray(x))
        h_offset = HexArray(x, "offset") * mersereau_region(HexArray(x, "offset"))
        shifted_oblique = fftshift(h_oblique)
        shifted_offset = fftshift(h_offset)
        assert np.abs(np.sum(h_oblique) - np.sum(shifted_oblique)) < 1e-12
        assert np.abs(np.sum(h_offset) - np.sum(shifted_offset)) < 1e-12

        hh_oblique = ifftshift(shifted_oblique)
        hh_offset = ifftshift(shifted_offset)
        assert np.allclose(h_oblique, hh_oblique, atol=1e-12)
        assert np.allclose(h_offset, hh_offset, atol=1e-12)


def test_hexarray():

    arr = np.ones((3, 3))

    # make sure the indices are in oblique coordinates by default
    # this corresponds to a 3x3 parallopiped
    hx = HexArray(arr)
    n1, n2 = hx.indices
    t1, t2 = np.meshgrid(np.arange(3), np.arange(3))
    assert np.all(n1 == t1)
    assert np.all(n2 == t2)

    # make sure internal representation in oblique coords is correct
    # when given an array with offset coordinates
    hx = HexArray(arr, pattern="offset")
    n1, n2 = hx.indices
    t1, t2 = np.meshgrid(np.arange(3), np.arange(3))

    # the row coordinates remain the same
    assert np.all(n1 == t1)
    # the column coordinates however...
    col_indices = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3]])
    assert np.all(n2 == col_indices)

    # test the pattern
    arr = np.ones((4, 5))
    hx = HexArray(arr, pattern="offset")
    n1, n2 = hx.indices
    col_indices = np.array([[i, i + 1, i + 1, i + 2] for i in range(5)])
    assert np.all(n2 == col_indices)


def test_rect_shift():
    # test with 4x3 shape
    N1, N2 = 4, 3
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
    data = np.sin(np.sqrt(n1**2 + n2**2)).T

    h = HexArray(data, pattern="offset")
    shifted = rect_shift(h)

    """
      *   o   o
    *   *   o
      *   *   o
    *   *   *

    o   o   *
      o   *   *
        o   *   *
           *   *   *
    """

    # point by point test
    assert h[1, 2] == shifted[1, 0]
    assert h[2, 2] == shifted[2, 0]
    assert h[3, 2] == shifted[3, 1]
    assert h[3, 1] == shifted[3, 0]

    assert np.abs(np.sum(h) - np.sum(shifted)) < 1e-12
    # test reverse
    assert np.allclose(h, rect_unshift(shifted))
