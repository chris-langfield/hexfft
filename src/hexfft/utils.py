import numpy as np
from hexfft.array import HexArray, _generate_indices
from hexfft.grids import heshgrid, skew_heshgrid


def hsupport(N, pattern="oblique"):
    """
    Return a boolean mask of the hexagonally periodic
    region inscribed in the square or parallelopiped region
    defined by h. h must have square dimensions with even
    side length. If h is NxN then the area of the mask
    is always 3 * N**2 / 4

    :param h: a HexArray
    """
    # assert N1 % 2 == 0, "Side length must be even."
    n1, n2 = _generate_indices((N, N), pattern)
    M = N // 2
    if pattern == "offset":
        n2 = n2 - N // 4

    G = (0 <= n1) & (n1 < 2 * M)
    H = (0 <= n2) & (n2 < 2 * M)
    I = (-M <= n1 - n2) & (n1 - n2 < M)
    condm = G & H & I
    mreg = condm.astype(float)
    assert np.sum(mreg) == 3 * M**2

    if pattern == "offset":
        mreg = np.flip(mreg.T)

    return mreg


def hex_to_pgram(h):
    """
    h is a square hexarray grid in oblique coordinates
    assuming the signal in h has periodic support on the corresponding
    Mersereau region, rearrange onto the equivalent parallelogram
    shaped region (in oblique coordinates).
    If h is NxN, then the size of the parallelogram
    will be (N//2, 3*(N//2))
    """
    # compute grid indices for hexagon in oblique coords
    N = h.shape[-1]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
    support = hsupport(N, h.pattern)
    # compute grid indices for parallelogram
    P = N // 2
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))

    # indices of the two halves of the hexagon
    # which be rearranged into a parallelogram
    support_below = support.astype(bool) & (n2 < P)
    support_above = support.astype(bool) & (n2 >= P)

    # corresponding chunks of parallelogram
    pgram_left = p2 > p1 - P
    pgram_right = p2 <= p1 - P

    if h.ndim == 3:
        nstack = h.shape[0]
        p = np.zeros((nstack, P, 3 * P), h.dtype)
        p[:, pgram_left] = h[:, support_below]
        p[:, pgram_right] = h[:, support_above]

    else:
        p = np.zeros((P, 3 * P), h.dtype)
        p[pgram_left] = h[support_below]
        p[pgram_right] = h[support_above]

    return HexArray(p, h.pattern)


def pgram_to_hex(p, N, pattern="oblique"):
    """
    N is required because there is ambiguity since P=N//2 - 1
    """
    if p.ndim == 3:
        nstack, P, P3 = p.shape
    else:
        P, P3 = p.shape
    assert P == N // 2
    assert P3 == 3 * P

    # compute grid indices for hexagon
    support = hsupport(N, pattern)
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))

    # compute grid indices for parallelogram
    p1, p2 = np.meshgrid(np.arange(3 * P), np.arange(P))

    # indices of the two halves of the hexagon
    # which be rearranged into a parallelogram
    support_below = support.astype(bool) & (n2 < P)
    support_above = support.astype(bool) & (n2 >= P)

    # corresponding chunks of parallelogram
    pgram_left = p2 > p1 - P
    pgram_right = p2 <= p1 - P

    if p.ndim == 3:
        h = HexArray(np.zeros((nstack, N, N), p.dtype), pattern)
        h[:, support_below] = p[:, pgram_left]
        h[:, support_above] = p[:, pgram_right]
    else:
        h = HexArray(np.zeros((N, N), p.dtype), pattern)
        h[support_below] = p[pgram_left]
        h[support_above] = p[pgram_right]

    return HexArray(h, pattern)


def nice_test_function(shape, hcrop=True, pattern="oblique"):
    h = HexArray(np.zeros(shape), pattern)
    N1, N2 = shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
    if hcrop:
        m = hsupport(N1, pattern)
    else:
        m = 1.0
    h[:, :] = (np.cos(n1) + 2 * np.sin((n1 - n2) / 4)) * m
    return h


def complex_type(type):
    if type in [np.float32, np.complex64]:
        return np.complex64
    else:
        return np.complex128
