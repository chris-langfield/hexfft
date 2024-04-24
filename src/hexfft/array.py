import numpy as np
from hexfft.grids import heshgrid, skew_heshgrid


def _generate_indices(shape, pattern):
    N1, N2 = shape
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))

    # convert offset indices to oblique for internal representation
    if pattern == "offset":
        row_shift = np.repeat(np.arange(N1), 2)[1 : N1 + 1]
        n2 += row_shift

    return n1, n2


def _generate_grid(shape, pattern):

    if pattern == "oblique":
        return skew_heshgrid(shape)
    elif pattern == "offset":
        return heshgrid(shape)


class HexArray(np.ndarray):
    """
    Wrapper for a NumPy array that can handle data sampled
    with oblique (slanted y-axis) or offset coordinates. Internally,
    offset coordinates are transformed to oblique.

    When pattern = "offset" by convention the origin is shifted
    to the left, so that the second row is to the right, the third
    row is in line with the first row, etc:

    row 0:  *   *   *   *
    row 1:    *   *   *   *
    row 2:  *   *   *   *
    row 3:    *   *   *   *

    ...

    """

    def __new__(cls, arr, pattern="oblique"):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(arr).view(cls)
        # add the new attribute to the created instance
        assert arr.ndim in [2, 3], "HexArray can only be of dimension 2 or 3."
        obj.pattern = pattern
        obj.indices = _generate_indices(arr.shape[-2:], pattern)
        obj.grid = _generate_grid(arr.shape[-2:], pattern)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.pattern = getattr(obj, "pattern", None)
        self.indices = getattr(obj, "indices", None)
        self.grid = getattr(obj, "grid", None)


def rect_shift(hx):
    """
    Shift a rectangular periodic region of support
    to a parallelogram for the hexfft with rectangular
    periodicity. See Ehrhardt eqn (9) and fig (4)

    :param hx: a HexArray with "offset" coordinates.
    :return: a HexArray with "oblique" coordinates with the data
        from hx shifted onto the parallelogram region of support.
    """
    # oblique coordinates
    f1, f2 = hx.indices

    # oblique coordinates of new region
    N1, N2 = hx.shape[-2:]
    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))

    # slice from rectangular region to shift
    upper_triangle = f2 >= hx.shape[-1]

    # slice of parallelogram region to transplant the upper triangle
    left_corner = n1 >= 2 * n2 + 1

    # transplant slice
    out = np.zeros(hx.shape, hx.dtype)
    if hx.ndim == 2:
        out[left_corner.T] = hx[upper_triangle.T]
        out[~left_corner.T] = hx[~upper_triangle.T]
    elif hx.ndim == 3:
        out[:, left_corner.T] = hx[:, upper_triangle.T]
        out[:, ~left_corner.T] = hx[:, ~upper_triangle.T]

    return HexArray(out, pattern="oblique")


def rect_unshift(hx):
    """
    UnShift a rectangular periodic region of support
    to a parallelogram for the hexfft with rectangular
    periodicity. See Ehrhardt eqn (9) and fig (4)

    :param hx: a HexArray with "oblique" coordinates.
    :return: a HexArray with "offset" coordinates with the data
        from hx shifted onto the parallelogram region of support.
    """
    out = HexArray(np.zeros(hx.shape, hx.dtype), "offset")

    # oblique coordinates
    n1, n2 = hx.indices

    # oblique coordinates of new region
    f1, f2 = out.indices

    # slice from rectangular region to shift
    upper_triangle = f2 >= hx.shape[-1]

    # slice of parallelogram region to transplant the upper triangle
    left_corner = n1 >= 2 * n2 + 1

    # transplant slice
    if hx.ndim == 2:
        out[upper_triangle.T] = hx[left_corner.T]
        out[~upper_triangle.T] = hx[~left_corner.T]
    elif hx.ndim == 3:
        out[:, upper_triangle.T] = hx[:, left_corner.T]
        out[:, ~upper_triangle.T] = hx[:, ~left_corner.T]
    
    return out
