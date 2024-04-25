import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from hexfft.utils import heshgrid, skew_heshgrid


def hexshow(h, ax=None, **args):
    N1, N2 = h.shape
    pattern = h.pattern
    assert np.isrealobj(h), f"Arrays of dtype {h.dtype} not supported for hexshow()"
    if pattern == "oblique":
        x1, x2 = skew_heshgrid(
            (N1, N2),
            matrix=np.array([[1, 0], [-1 / 2, 1]]),
        )
    elif pattern == "offset":
        x1, x2 = heshgrid((N1, N2))

    patches = []
    for i in range(N1):
        for j in range(N2):
            _x1, _x2 = x1[i, j], x2[i, j]
            polygon = mpatches.RegularPolygon([_x1, _x2], 6, radius=0.6, linewidth=0)
            patches.append(polygon)

    c = PatchCollection(patches, alpha=1.0, match_original=True)

    if ax is None:
        fig, ax = plt.subplots()
    ax.add_collection(c)
    c.set(array=h.ravel(), **args)
    ax.set_xlim(0, N2)
    ax.set_ylim(0, N1)
    ax.axis("equal")
    return c


class ObliqueArray:
    def __init__(self, shape, matrix=None, dtype=np.float64):
        if matrix is None:
            # regular hexagonal grid
            matrix = np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
        self.shape = shape
        self.matrix = matrix
        self.dtype = dtype

    def meshgrid(self):
        return skew_heshgrid(self.shape, self.matrix, self.dtype)

    @property
    def indices(self):
        return np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))

    def to_cartesian(self, n1, n2):
        """
        Given a pair of coordinates (n1, n2) return
        the cartesian sample point (relative to (0, 0))
        """
        return self.matrix.T @ np.array([n1, n2])

    def get_spacing(self):
        # norms of basis vectors
        d1 = np.sqrt(np.sum(self.matrix[0, :] ** 2))
        d2 = np.sqrt(np.sum(self.matrix[1, :] ** 2))
        return d1, d2

    def evaluate(self, f):
        """
        :param f: A function of X and Y coordinates
        """
        x, y = self.meshgrid()
        return f(x, y)

    def plot(self, z, ax=None, s=None, **args):
        if s is None:
            s = 1600 / (self.shape[0])
        args["s"] = s
        if ax is None:
            fig, ax = plt.subplots()
        ax.axis("equal")
        x, y = self.meshgrid()
        return ax.scatter(x, y, c=z, marker="h", **args)

    def hregion(self, center, size):
        """
        return mask for a hexagonal region of support
        with side length size centered at center
        """
        h1, h2 = center
        n1, n2 = self.indices
        A = n2 < h2 + size
        B = n2 > h2 - size
        C = n1 > h1 - size
        D = n1 < h2 + size
        E = n2 < n1 + (h2 - h1) + size
        F = n2 > n1 + (h2 - h1) - size
        cond = A & B & C & D & E & F
        return cond.astype(int)

    def default_hregion(self):
        n = self.shape[0]
        if n % 2 == 1:
            center = (n // 2, n // 2)
            size = n // 2 + 1
            return self.hregion(center, size)
        else:
            center = (n / 2 - 1, n / 2 - 1)
            size = n / 2
            return self.hregion(center, size)


class OffsetArray:
    def __init__(self, shape, dtype=np.float64):
        self.shape = shape
        self.dtype = dtype

    def meshgrid(self):
        return heshgrid(self.shape)

    def plot(self, z, ax=None, s=None, **args):
        if s is None:
            s = 1600 / (self.shape[0])
        args["s"] = s
        if ax is None:
            fig, ax = plt.subplots()
        ax.axis("equal")
        x, y = self.meshgrid()
        return ax.scatter(x, y, c=z, marker="h", **args)
