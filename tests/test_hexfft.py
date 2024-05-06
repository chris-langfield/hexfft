from hexfft import fftshift, ifftshift, HexArray
from hexfft.hexfft import (
    _hexdft_pgram,
    _hexidft_pgram,
    _hexdft_pgram_stack,
    _hexidft_pgram_stack,
    mersereau_fft,
    mersereau_ifft,
    rect_fft,
    rect_ifft,
    FFT,
    fft,
    ifft,
)
from hexfft.utils import (
    hsupport,
    pgram_to_hex,
    nice_test_function,
    hex_to_pgram,
)
from hexfft.array import rect_shift, rect_unshift
from hexfft.reference import (
    _hexdft_slow,
    _hexidft_slow,
    _rect_dft_slow,
    _rect_idft_slow,
)
import numpy as np
import pytest


def hregion(n1, n2, center, size, pattern="oblique"):
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
    return HexArray(cond.astype(int), pattern)


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

        IMPULSE = _hexdft_slow(impulse)
        impulse_T = _hexidft_slow(IMPULSE)

        m = hsupport(N)

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

    # test stack
    nstack = 10
    for size in [5, 6, 20, 21, 50, 51]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        N = n1.shape[0]
        if N % 2 == 1:
            center = (N // 2, N // 2)
        else:
            center = (N / 2 - 1, N / 2 - 1)
        impulse = HexArray(
            np.stack([hregion(n1, n2, center, i + 1) for i in range(nstack)]), "oblique"
        )
        impulse_single = HexArray(hregion(n1, n2, center, 1), "oblique")

        impulse_p = hex_to_pgram(impulse)
        impulse_single_p = hex_to_pgram(impulse_single)

        IMPULSE_P = _hexdft_pgram_stack(impulse_p)
        IMPULSE_SINGLE_P = _hexdft_pgram(impulse_single_p)
        assert np.allclose(IMPULSE_P[0], IMPULSE_SINGLE_P)

        impulse_p_T = _hexidft_pgram_stack(IMPULSE_P)
        impulse_single_p_T = _hexidft_pgram(IMPULSE_SINGLE_P)
        assert np.allclose(impulse_p_T[0], impulse_single_p_T)

        assert np.allclose(impulse_p, impulse_p_T, atol=1e-12)


@pytest.mark.skip(reason="reference slow DFT implementation TODO")
def test_rect_dft():
    # test rect dft and idft
    for shape in [(4, 5), (8, 8), (11, 16), (19, 19)]:
        N1, N2 = shape
        # N2 must be even
        N2 = N1 + N1 % 2
        n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
        center = (N1 / 2 - 1, N2 / 2 - 1)
        d = hregion(n1, n2, center, 1)

        D = _rect_dft_slow(d)
        dd = _rect_idft_slow(D)
        assert np.allclose(d, dd, atol=1e-12)


@pytest.mark.skip(reason="reference slow DFT implementation TODO")
def test_rect_fft():
    for pattern in ["oblique", "offset"]:
        for shape in [(4, 5), (8, 8), (11, 16), (19, 19)]:
            N1, N2 = shape
            n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
            center = (N1 // 2 - 1, N2 // 2 - 1)
            d = rect_shift(hregion(n1, n2, center, 1, "offset"))

            D = rect_fft(d)
            D_slow = _rect_dft_slow(d)

            assert np.allclose(D, D_slow)

            dd = rect_ifft(D)
            dd_slow = _rect_idft_slow(D_slow)

            assert np.allclose(dd, dd_slow)


@pytest.mark.parametrize("shape", [(16, 16), (17, 17), (16, 17), (17, 16)])
@pytest.mark.parametrize("mode", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (5, 7)])
def test_validate_rect_fft(shape, mode):
    N1, N2 = shape
    k1, k2 = mode
    n1, n2 = np.meshgrid(np.arange(N2), np.arange(N1))
    fmode = HexArray(
        np.exp(2 * np.pi * 1.0j * (k1 * (n1 - n2 / 2) / N2 + k2 * n2 / N1)), "oblique"
    )
    FMODE = rect_fft(fmode)

    assert np.allclose(FMODE[k2, k1], N1 * N2)


def test_rect_fft_stack():
    # create a stack of data
    nstack, N1, N2 = 5, 8, 12
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
    center = (N1 / 2 - 1, N2 / 2 - 1)
    data = np.stack([hregion(n1, n2, center, i).T for i in range(1, nstack + 1)])
    data = HexArray(data, "offset")

    fftobj = FFT((N1, N2), periodicity="rect")
    DATA_STACK = fftobj.forward(data)

    data_shifted = rect_shift(data)
    DATA_LOOP = np.stack(
        [rect_unshift(rect_fft(data_shifted[i])) for i in range(nstack)]
    )
    DATA_LOOP = HexArray(DATA_LOOP, "offset")

    assert np.allclose(DATA_STACK, DATA_LOOP)

    ddata_stack = fftobj.inverse(DATA_STACK)
    ddata_loop = np.stack(
        [rect_unshift(rect_ifft(rect_shift(DATA_LOOP[i]))) for i in range(nstack)]
    )
    ddata_loop = HexArray(ddata_loop, "offset")

    assert np.allclose(ddata_stack, ddata_loop)

    # test singleton
    ds = data[0]
    DS_STACK = fftobj.forward(ds)
    DS_SINGLE = rect_unshift(rect_fft(rect_shift(ds)))
    assert np.allclose(DS_STACK, DS_SINGLE)

    dds_stack = fftobj.inverse(DS_STACK)
    dds_single = rect_unshift(rect_ifft(rect_shift(DS_SINGLE)))
    assert np.allclose(dds_stack, dds_single)


def test_rect_fft_stack_noshift():
    # create a stack of data
    nstack, N1, N2 = 5, 8, 12
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
    center = (N1 / 2 - 1, N2 / 2 - 1)
    data = np.stack([hregion(n1, n2, center, i).T for i in range(1, nstack + 1)])
    data = HexArray(data, "oblique")

    fftobj = FFT((N1, N2), periodicity="rect")
    DATA_STACK = fftobj.forward(data)

    DATA_LOOP = np.stack([rect_fft(data[i]) for i in range(nstack)])
    DATA_LOOP = HexArray(DATA_LOOP, "offset")

    assert np.allclose(DATA_STACK, DATA_LOOP)

    ddata_stack = fftobj.inverse(DATA_STACK)
    ddata_loop = np.stack([rect_ifft(DATA_LOOP[i]) for i in range(nstack)])
    ddata_loop = HexArray(ddata_loop, "offset")

    assert np.allclose(ddata_stack, ddata_loop)

    # test singleton
    ds = data[0]
    DS_STACK = fftobj.forward(ds)
    DS_SINGLE = rect_fft(ds)
    assert np.allclose(DS_STACK, DS_SINGLE)

    dds_stack = fftobj.inverse(DS_STACK)
    dds_single = rect_ifft(DS_SINGLE)
    assert np.allclose(dds_stack, dds_single)


def test_hex_fft_stack():
    # create a stack of data
    for pattern in ["oblique", "offset"]:
        for size in [4, 8, 16, 32]:
            nstack, N1, N2 = 5, size, size
            n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
            center = (size // 2, size // 2)
            data = np.stack([hregion(n1, n2, center, i) for i in range(1, nstack + 1)])
            data = HexArray(data, pattern)

            fftobj = FFT((size, size), periodicity="hex")

            DATA_STACK = fftobj.forward(data)
            DATA_LOOP = np.stack([fft(data[i], "hex") for i in range(nstack)])
            DATA_LOOP = HexArray(DATA_LOOP, pattern)

            assert np.allclose(DATA_STACK, DATA_LOOP)

            ddata_stack = fftobj.inverse(DATA_STACK)
            ddata_loop = np.stack([ifft(DATA_LOOP[i], "hex") for i in range(nstack)])
            ddata_loop = HexArray(ddata_loop, pattern)

            assert np.allclose(ddata_stack, ddata_loop)

            # test singleton
            ds = data[0]
            DS_STACK = fftobj.forward(ds)
            DS_SINGLE = fft(ds, "hex")
            assert np.allclose(DS_STACK, DS_SINGLE)

            dds_stack = fftobj.inverse(DS_STACK)
            dds_single = ifft(DS_SINGLE, "hex")
            assert np.allclose(dds_stack, dds_single)


def test_mersereau_fft():
    # testing in float64

    for size in [4, 8, 16, 32]:
        n1, n2 = np.meshgrid(np.arange(size), np.arange(size))
        N = n1.shape[0]
        center = (N / 2 - 1, N / 2 - 1)

        # test on an impulse function and on an arbitrary function
        # d for "dirac"
        d = hregion(n1, n2, center, 1)
        x = nice_test_function((N, N))
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
        x = nice_test_function((size, size))
        h_oblique = HexArray(x, "oblique") * hsupport(size)
        h_offset = HexArray(x, "offset") * hsupport(size, "offset")
        shifted_oblique = fftshift(h_oblique)
        shifted_offset = fftshift(h_offset)
        assert np.abs(np.sum(h_oblique) - np.sum(shifted_oblique)) < 1e-12
        assert np.abs(np.sum(h_offset) - np.sum(shifted_offset)) < 1e-12

        hh_oblique = ifftshift(shifted_oblique)
        hh_offset = ifftshift(shifted_offset)
        assert np.allclose(h_oblique, hh_oblique, atol=1e-12)
        assert np.allclose(h_offset, hh_offset, atol=1e-12)
