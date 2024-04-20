import numpy as np
from hexfft.array import HexArray


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
    #assert N1 % 2 == 0, "Side length must be even."
    n1, n2 = h._indices
    M = N1 // 2
    if h.pattern == "offset":
        n2 = n2 - N1 // 4

    G = (0 <= n1) & (n1 < 2 * M)
    H = (0 <= n2) & (n2 < 2 * M)
    I = (-M <= n1 - n2) & (n1 - n2 < M)
    condm = G & H & I
    mreg = condm.astype(float)
    assert np.sum(mreg) == 3 * M ** 2

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
    p1, p2 = np.meshgrid(np.arange(3*P), np.arange(P))

    # indices of the two halves of the hexagon
    # which be rearranged into a parallelogram
    support_below = support.astype(bool) & (n2 < P)
    support_above = support.astype(bool) & (n2 >= P)

    # corresponding chunks of parallelogram
    pgram_left = p2 > p1 - P
    pgram_right = p2 <= p1 - P

    p = np.zeros((P, 3*P), h.dtype)
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
    p1, p2 = np.meshgrid(np.arange(3*P), np.arange(P))

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
    P = x.shape[0] # i.e. = N
    # Parallelogram (square in oblique coordinates) enclosing
    M = 2*(P+1)
    m1, m2 = np.meshgrid(np.arange(M), np.arange(M))
    grid = np.zeros((M, M), x.dtype)
    grid[int(P//2):P + int(P//2), int(P//2):P + int(P//2)] = x

    return grid
    

def heshgrid(shape, t=(1., 1.), dtype=np.float64):
    """

    Returns coordinates for a *regularly* hexagonally sampled rectangular
    region with shape = (x, y) samples in the x and y directions respectively.

    t = (t1, t2) are the sampling lengths between x and y points respectively

    The x and y bounds are (0, t1*(x+1/2)) and (0, t2*y) respectively.

    x, y = heshgrid((10,10), (1,1))
    plt.scatter(x,y)

    :param shape: (nrows, ncols) 
    :param t: (t1, t2) sample distance in x and y
    :return: x and y grid points for hexagonal sampling
    of the region t1*nr x t2*nc
    """
    nr, nc = shape
    t1, t2 = t
    x0, y0 = np.meshgrid(np.arange(0, t1*nc, t1), np.arange(0, t2*nr, 2*t2))
    x1, y1 = np.meshgrid(np.arange(t1/2, t1*nc + t1/2, t1), np.arange(t2, t2*nr + t2, 2*t2))
    outx = np.zeros((nr, nc), dtype)
    outy = np.zeros((nr, nc), dtype)
    outx[::2, :] = x0
    outx[1::2, :] = x1
    outy[::2, :] = y0
    outy[1::2, :] = y1
    return outx, outy

def skew_heshgrid(shape, matrix=None, dtype=np.float64):
    """

    Returns coordinates for a hexagonally sampled rhomboid
    region with shape = (x, y) samples in the x and y directions respectively.

    matrix is the sampling matrix containing the basis of R^2 used for the tiling.
    i.e. matrix = [b0, b1]. Note that if b0 has the form [r,0] and b1 has the form [s/2, s],
    this is non-skewed hexagonal grid.

    If s = 2r/sqrt3 this is regular hexagonal sampling.

    :param shape: (nrows, ncols)
    :param matrix: sampling matrix 2x2 where the rows
        form a basis for R^2
    """
    if matrix is None:
        # regular hexagonal grid
        matrix = np.array([[1,0], [-1/2, np.sqrt(3)/2]])
    nr, nc = shape
    b0 = matrix[0, :]
    b1 = matrix[1, :]
    # make sure the matrix has rank 2
    assert np.linalg.det(matrix) != 0, "Basis vectors are not linearly independent"
    x = np.tile(np.arange(nc), (nr,1))
    shiftx = np.arange(nr) * b1[0]
    x = (x + np.tile(shiftx, (nc, 1)).T) * b0[0]
    y = np.tile(np.arange(nr), (nc, 1)).T
    shifty = np.arange(nc) * b0[1]
    y = (y + np.tile(shifty, (nr, 1))) * b1[1]

    return x, y

def nice_test_function(n1, n2, mersereau=True):
    h = HexArray(np.zeros(n1.shape), pattern="oblique")
    if mersereau:
        m = mersereau_region(h)
    else:
        m = 1.
    h[:,:] = (np.cos(n1) + 2 * np.sin((n1 - n2)/4)) * m
    return h
