import numpy as np


def heshgrid(shape, t=(1.0, 1.0), dtype=np.float64):
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
    x0, y0 = np.meshgrid(np.arange(0, t1 * nc, t1), np.arange(0, t2 * nr, 2 * t2))
    x1, y1 = np.meshgrid(
        np.arange(t1 / 2, t1 * nc + t1 / 2, t1), np.arange(t2, t2 * nr + t2, 2 * t2)
    )
    outx = np.zeros((nr, nc), dtype)
    outy = np.zeros((nr, nc), dtype)
    outx[::2, :] = x0
    outx[1::2, :] = x1[: nr // 2, :]
    outy[::2, :] = y0
    outy[1::2, :] = y1[: nr // 2, :]
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
        matrix = np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
    nr, nc = shape
    b0 = matrix[0, :]
    b1 = matrix[1, :]
    # make sure the matrix has rank 2
    assert np.linalg.det(matrix) != 0, "Basis vectors are not linearly independent"
    x = np.tile(np.arange(nc), (nr, 1))
    shiftx = np.arange(nr) * b1[0]
    x = (x + np.tile(shiftx, (nc, 1)).T) * b0[0]
    y = np.tile(np.arange(nr), (nc, 1)).T
    shifty = np.arange(nc) * b0[1]
    y = (y + np.tile(shifty, (nr, 1))) * b1[1]

    return x, y
