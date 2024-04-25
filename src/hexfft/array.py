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

    def __new__(cls, arr, pattern="offset"):
        # np.ndarray subclass boilerplate
        obj = np.asarray(arr).view(cls)

        if arr.ndim not in [2, 3]:
            raise ValueError("HexArray can only be of dimension 2 or 3.")

        if pattern not in ["oblique", "offset"]:
            raise ValueError(
                f"Coordinate system {pattern} is not implemented or unknown."
            )

        obj.pattern = pattern
        obj.indices = _generate_indices(arr.shape[-2:], pattern)
        obj.grid = _generate_grid(arr.shape[-2:], pattern)

        return obj

    # np.ndarray subclass boilerplate
    def __array_finalize__(self, obj):
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
    assert hx.pattern == "offset", "Array must be in 'offset' coordinates."

    N1, N2 = hx.shape[-2:]
    if N1 < 2 or N2 < 2:
        raise ValueError("Cannot shift offset array of size less than 2x2.")

    n1, n2 = np.meshgrid(np.arange(N1), np.arange(N2))

    # oblique coordinates
    f1, f2 = hx.indices

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
    assert hx.pattern == "oblique", "Array must be in 'oblique' coordinates."

    N1, N2 = hx.shape[-2:]
    if N1 < 2 or N2 < 2:
        raise ValueError("Cannot shift offset array of size less than 2x2.")
    # oblique coordinates
    n1, n2 = hx.indices

    out = HexArray(np.zeros(hx.shape, hx.dtype), "offset")

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
