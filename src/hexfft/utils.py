import numpy as np
from hexfft.array import HexArray
from hexfft.grids import heshgrid, skew_heshgrid


def mersereau_region(h):
    """
    Return a boolean mask of the hexagonally periodic
    region inscribed in the square or parallelopiped region
    defined by h. h must have square dimensions with even
    side length. If h is NxN then the area of the mask
    is always 3 * N**2 / 4

    :param h: a HexArray
    """
    N1, N2 = h.shape
    assert N1 == N2, "Only square arrays are allowed."
    # assert N1 % 2 == 0, "Side length must be even."
    n1, n2 = h.indices
    M = N1 // 2
    if h.pattern == "offset":
        n2 = n2 - N1 // 4

    G = (0 <= n1) & (n1 < 2 * M)
    H = (0 <= n2) & (n2 < 2 * M)
    I = (-M <= n1 - n2) & (n1 - n2 < M)
    condm = G & H & I
    mreg = condm.astype(float)
    assert np.sum(mreg) == 3 * M**2

    if h.pattern == "offset":
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
    N = h.shape[0]
    n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
    support = mersereau_region(h)
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

    p = np.zeros((P, 3 * P), h.dtype)
    p[pgram_left] = h[support_below]
    p[pgram_right] = h[support_above]

    return HexArray(p, pattern=h.pattern)


def pgram_to_hex(p, N, pattern="oblique"):
    """
    N is required because there is ambiguity since P=N//2 - 1
    """
    P = p.shape[0]
    assert P == N // 2

    # compute grid indices for hexagon
    h = HexArray(np.zeros((N, N), p.dtype), pattern=pattern)
    support = mersereau_region(h)
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

    h[support_below] = p[pgram_left]
    h[support_above] = p[pgram_right]

    return HexArray(h, pattern=h.pattern)


def pad(x):
    """
    Given an NxN array x, find the enclosing Mersereau
    hexagonal region and sampling grid.
    """
    assert x.shape[0] == x.shape[1]

    # Create a Mersereau hexagonal region of size N
    P = x.shape[0]  # i.e. = N
    # Parallelogram (square in oblique coordinates) enclosing
    M = 2 * (P + 1)
    m1, m2 = np.meshgrid(np.arange(M), np.arange(M))
    grid = np.zeros((M, M), x.dtype)
    grid[int(P // 2) : P + int(P // 2), int(P // 2) : P + int(P // 2)] = x

    return grid


def nice_test_function(n1, n2, mersereau=True):
    h = HexArray(np.zeros(n1.shape), pattern="oblique")
    if mersereau:
        m = mersereau_region(h)
    else:
        m = 1.0
    h[:, :] = (np.cos(n1) + 2 * np.sin((n1 - n2) / 4)) * m
    return h
