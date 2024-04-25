import numpy as np
from hexfft import HexArray
from hexfft.array import rect_shift, rect_unshift
from hexfft.utils import pgram_to_hex, hex_to_pgram


def test_hexarray():
    # test stack and single image
    arrs = [np.ones((3, 3)), np.ones((10, 3, 3))]
    for arr in arrs:
        # make sure the indices are in oblique coordinates by default
        # this corresponds to a 3x3 parallopiped
        hx = HexArray(arr, "oblique")
        n1, n2 = hx.indices
        t1, t2 = np.meshgrid(np.arange(3), np.arange(3))
        assert np.all(n1 == t1)
        assert np.all(n2 == t2)

        # make sure internal representation in oblique coords is correct
        # when given an array with offset coordinates
        hx = HexArray(arr, "offset")
        n1, n2 = hx.indices
        t1, t2 = np.meshgrid(np.arange(3), np.arange(3))

        # the row coordinates remain the same
        assert np.all(n1 == t1)
        # the column coordinates however...
        col_indices = np.array([[0, 1, 1], [1, 2, 2], [2, 3, 3]])
        assert np.all(n2 == col_indices)

        # test the pattern
        arr = np.ones((4, 5))
        hx = HexArray(arr, "offset")
        n1, n2 = hx.indices
        col_indices = np.array([[i, i + 1, i + 1, i + 2] for i in range(5)])
        assert np.all(n2 == col_indices)


def test_hex_pgram_conversions():
    # all in oblique coordinates
    nstack = 10
    for N in [5, 8, 16, 21]:
        pgram = HexArray(np.random.normal(size=(N // 2, 3 * (N // 2))), "oblique")
        pgrams = HexArray(np.stack([pgram * (i + 1) for i in range(nstack)]), "oblique")

        hex = pgram_to_hex(pgram, N)
        hexs = pgram_to_hex(pgrams, N)

        assert np.all(hexs[0] == hex)

        ppgram = hex_to_pgram(hex)
        ppgrams = hex_to_pgram(hexs)

        assert np.all(ppgram == pgram)
        assert np.all(ppgrams == pgrams)


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


def test_rect_shift_stack():
    # test with 4x3 shape
    N1, N2 = 4, 3
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))
    data = np.stack([np.sin(np.sqrt(n1**2 + n2**2)).T + i for i in range(10)])
    h = HexArray(data, pattern="offset")
    shifted = rect_shift(h)
    # point by point along stack axis
    assert np.all(h[:, 1, 2] == shifted[:, 1, 0])
    assert np.all(h[:, 2, 2] == shifted[:, 2, 0])
    assert np.all(h[:, 3, 2] == shifted[:, 3, 1])
    assert np.all(h[:, 3, 1] == shifted[:, 3, 0])

    assert np.abs(np.sum(h) - np.sum(shifted)) < 1e-12
    # test reverse
    assert np.allclose(h, rect_unshift(shifted))
