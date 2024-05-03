from hexfft import HexArray, fft, ifft, FFT, fftshift, ifftshift
from hexfft.array import rect_shift, rect_unshift
import numpy as np
import pytest

array = np.random.normal(size=(16, 16)).astype(np.float64)
patterns = ["oblique", "offset"]
h_obl = HexArray(array, "oblique")
h_off = HexArray(array, "offset")

periodicities = ["rect", "hex"]


@pytest.mark.parametrize("pattern", patterns)
def test_HexArray_initialization(pattern):
    array4 = np.random.normal(size=(4, 4, 4, 4))

    harray = HexArray(array, pattern)
    assert harray.dtype == array.dtype
    assert harray.shape == array.shape
    assert np.all(harray[:5, :2] == array[:5, :2])

    with pytest.raises(ValueError, match="dimension 2 or 3"):
        harray = HexArray(array[0, :], pattern)

    with pytest.raises(ValueError, match="dimension 2 or 3"):
        harray = HexArray(array4, pattern)


def test_HexArray_known_patterns():
    for p in patterns:
        h = HexArray(array, p)

    with pytest.raises(ValueError, match="not implemented or unknown"):
        _ = HexArray(array, "bla")


def test_rect_shift_pathological_inputs():
    with pytest.raises(AssertionError, match="must be in 'offset'"):
        _ = rect_shift(h_obl)

    with pytest.raises(AssertionError, match="must be in 'oblique'"):
        _ = rect_unshift(h_off)

    with pytest.raises(ValueError, match="less than 2x2"):
        _ = rect_shift(h_off[:2, :1])

    with pytest.raises(ValueError, match="less than 2x2"):
        _ = rect_unshift(h_obl[:2, :1])


def test_fft_call():
    with pytest.raises(ValueError, match="Unrecognized periodicity"):
        _ = fft(array, "bla")

    with pytest.raises(ValueError, match="Unrecognized periodicity"):
        _ = ifft(array, "bla")

    # forbidden Arrays for hexagonal periodicity
    with pytest.raises(AssertionError, match="square"):  # hex fft input must be square
        _ = fft(array[:4, :], "hex")

    with pytest.raises(
        AssertionError, match="multiple of 4."
    ):  # hex fft input must be multiple of 4
        _ = fft(array[:10, :10], "hex")

    with pytest.raises(AssertionError, match="square"):  # hex ifft input must be square
        _ = ifft(array[:4, :], "hex")

    with pytest.raises(
        AssertionError, match="multiple of 4."
    ):  # hex ifft input must be multiple of 4
        _ = ifft(array[:10, :10], "hex")


@pytest.mark.parametrize("per", periodicities)
def test_FFT_constr(per):
    with pytest.raises(AssertionError, match="dtype of transform"):
        _ = FFT((8, 8), per, np.float32)
    with pytest.raises(AssertionError, match="Only 2D transforms"):
        _ = FFT((8, 8, 8), per)


def test_FFT_hex_constr():
    with pytest.raises(
        AssertionError, match="Input to FFT with hex periodicity must be a square array"
    ):
        _ = FFT((8, 4), "hex")

    with pytest.raises(
        AssertionError,
        match="Array size for periodicity='hex' must be a multiple of 4.",
    ):
        _ = FFT((10, 10), "hex")


@pytest.mark.parametrize("pattern", patterns)
def test_hex_fft_output(pattern):
    X = fft(HexArray(array, pattern), "hex")
    assert X.shape == array.shape
    assert X.dtype == np.complex128

    X = fft(HexArray(array.astype(np.float32), pattern), "hex")
    assert X.dtype == np.complex64

    X = fft(HexArray(array.astype(np.complex64), pattern), "hex")
    assert X.dtype == np.complex64

    X = fft(HexArray(array.astype(np.complex128), pattern), "hex")
    assert X.dtype == np.complex128


def test_rect_fft_output():
    X = fft(HexArray(array, "offset"), "rect")
    assert X.shape == array.shape
    assert X.dtype == np.complex128

    # standard 1D fft generally not supported for single precision

    # X = fft(HexArray(array.astype(np.float32), "offset"), "rect")
    # assert X.dtype == np.complex64

    # X = fft(HexArray(array.astype(np.complex64), "offset"), "rect")
    # assert X.dtype == np.complex64

    X = fft(HexArray(array.astype(np.complex128), "offset"), "rect")
    assert X.dtype == np.complex128
