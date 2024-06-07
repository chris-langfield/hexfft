import numpy as np
import matplotlib.pyplot as plt
import scipy

import pygsp
from pygsp import utils

from hexfft import HexArray, ifft
from hexfft.plot import hexshow
from hexfft.array import generate_grid

# ---------------------------------------------------
W = np.array([[0., 0., 1., 1., 0.],
       [0., 0., 1., 1., 1.],
       [1., 1., 0., 0., 1.],
       [1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0.]])

g = pygsp.graphs.Graph(W)
g.set_coordinates()
g.plot_signal(np.ones(5))


f = np.random.randn(5)
print(f)
fig, ax = plt.subplots()
g.plot_signal(f, ax=ax)
ax.set_title("Random signal")

# ---------------------------------------------------
# 1D (ring graph)

N = 32
rg = pygsp.graphs.Ring(N)
sine = np.sin(2*np.pi*np.arange(N)/N)
rg.plot_signal(sine)

rg.compute_fourier_basis()
fig, ax = plt.subplots()
rg.plot_signal(rg.U[:, 3], ax=ax)
ax.set_title("3rd graph Laplacian eigenvector")

# adjacency matrix
fig, ax = plt.subplots()
ax.matshow(rg.W.toarray())
ax.set_title("N=32 Ring graph adjacency matrix")

fig, axs = plt.subplots(4, 8)
for i, ax in enumerate(axs.flat):
    ax.plot(rg.U[:, i])
fig.suptitle("N=32 Ring Graph: Eigenvector modes")

Uf = np.zeros((N, N), np.complex128)
for i in range(N):
    Uf[:, i] = np.exp(-2.j * np.pi * i * np.arange(N) / N)
fig, axs = plt.subplots(4, 8)
for i, ax in enumerate(axs.flat):
    ax.plot(Uf[:, i], "red")
fig.suptitle("N=32 Ring Graph: Fourier modes")

fig, axs = plt.subplots(6, 3)
axs[0, 0].plot(np.sqrt(32)*rg.U[:, 0])
axs[0, 0].set_yticks([0.95, 1, 1.05])
axs[0, 2].plot(Uf[:, 0], "red")
axs[0, 0].set_title("Eigenvector modes", color="blue")
axs[0, 2].set_title("Fourier modes (real part)", color="red")
for i in range(5):
    axs[i+1, 0].plot(4*rg.U[:, 2*i+1])
    axs[i+1, 1].plot(4*rg.U[:, 2*i + 2])
    axs[i+1, 2].plot(Uf[:, i+1], "red")
for i, ax in enumerate(axs.flat):
    ax.set_xticks([0, 16])
    if i % 3 ==1:
        ax.set_yticklabels([])
        ax.yaxis.set_visible(False)
    if i % 3 == 2:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"            k={i//3}", rotation=0,fontdict=dict(weight='bold'))
    if i not in [17, 16, 15]:
        ax.set_xticklabels([])

fig.suptitle("Ring graph: Graph Fourier basis vs 1D Fourier basis\n\n")

# eigenvalues
eigs = np.sort(np.linalg.eigvals(rg.L.toarray()))
eigs_analytic = np.sort(np.array([2*(1-np.cos(2*np.pi * k / N)) for k in range(N)]))

fig, ax = plt.subplots()
ax.plot(np.sort(eigs)[::-1])
ax.set_ylabel("λi    ", rotation=0, fontsize=16)
ax.set_xlabel("i", fontsize=16)
fig.suptitle("Graph Laplacian Eigenvalues: N=32 Ring Graph")


# ---------------------------------------------------
# 2D (square graph)

N1, N2 = 12, 12
sg = pygsp.graphs.Grid2d(N1, N2)
sg.plot_signal(np.ones(N1*N2))

fig, ax = plt.subplots()
ax.matshow(sg.W.toarray())
ax.set_title("Adjacency matrix for 6x6 square lattice graph")

N1, N2 = 12, 12
circ = np.zeros(N1*N2)
circ[[1, N2, N1*N2-N2, N1*N2-1]] = 1
adj = scipy.linalg.circulant(circ).T

psg = pygsp.graphs.Graph(W=adj)
psg.set_coordinates()
fig, ax = plt.subplots()
psg.plot_signal(np.ones(N1*N2), ax=ax)
ax.set_title("6x6 toroid graph (square lattice with periodic boundary conditions)")

fig, ax = plt.subplots()
ax.matshow(adj)
ax.set_title("Adjacency matrix for 6x6 square lattice graph\n[periodic boundary conditions]")

sg.compute_fourier_basis()

analytic_U = np.zeros((N1*N2, N1*N2))
for k in range((N1*N2)):
    analytic_U[:, k] = np.exp(2.j * np.pi * k * np.arange(N1*N2)/(N1*N2))

fig, axs = plt.subplots(4, 4)
for i, ax in enumerate(axs.flat):
    ax.matshow(sg.U[:, i].reshape(N1, N2))
fig.suptitle("12x12 Square Graph: Eigenvector modes (no periodic boundary conditions)")

Uf = np.zeros((N1, N2, N1, N2), np.complex128)
x, y = np.meshgrid(np.arange(N1), np.arange(N2), indexing="xy")
for i in range(N1):
    for j in range(N2):
        Uf[i, j, :, :] = np.exp(2.j * np.pi * (i*x/N1+j*y/N2))
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i, j].matshow(np.real(Uf[i, j, :, :]))
fig.suptitle("12x12 Square Graph: Fourier modes")

eigs_noboundary = np.linalg.eigvals(sg.L.toarray())
eigs_boundary = np.linalg.eigvals(psg.L.toarray())
eigs_analytic = np.sort(
    np.array(
        [4-2*np.cos(2*np.pi*k/(N1**2)) - 2*np.cos(2*np.pi*k/N1) 
         for k in range(N1**2)]
    )
)

fig, ax = plt.subplots()
ax.plot(np.sort(eigs_noboundary)[::-1], label="numerical - no boundary conditions")
ax.plot(np.sort(eigs_boundary)[::-1], label="numerical - with periodic boundary conditions")
ax.plot(np.sort(eigs_analytic)[::-1], label="analytic - with periodic boundary conditions")
fig.suptitle("Graph Laplacian eigenvalues: 12x12 Square Grid Graph")
ax.legend()

# ---------------------------------------------------
# 2D (triangle graph)


def TriangleGrid2d(N1, N2, periodic=True):
    W = np.zeros((N1*N2, N1*N2))

    for i in range(N1*N2-1):
        j = i + 1
        if j % N2 == 0:
            continue
        else:
            W[i, j] = 1.
            W[j, i] = 1.

    for i in range(N1):
        if i == 0:
            W[i, N2] = 1.
            for j in range(1, N2):
                W[i + j, N2 + j - 1: N2 + j + 1] = 1.
        elif i == N1 - 1 and i % 2 == 0:
            W[i*N2, i*N2 - N2] = 1.
            for j in range(1, N2):
                W[i*N2 + j, -N2 + i*N2 + j - 1: -N2 + i*N2 + j + 1] = 1.
        elif i % 2 == 0:
            W[i*N2, i*N2 + N2] = 1.
            W[i*N2, i*N2 - N2] = 1.
            for j in range(1, N2):
                W[i*N2 + j, N2 + i*N2 + j - 1: N2 + i*N2 + j + 1] = 1.
                W[i*N2 + j, -N2 + i*N2 + j - 1: -N2 + i*N2 + j + 1] = 1.
        
    if periodic:
        for i in range(N1):
            # side to side
            W[i*N2, (i+1)*N2 - 1] = 1.
            # slant right connections at sides
            if i % 2 == 0:
                W[i*N2, (i+2)*N2 -1] = 1.
                # slant left connections at sides
                W[i*N2-1, i*N2] = 1.
        for i in range(N2):
        # slant right connections at top
            W[i, N1*N2 - N2 + i] = 1.
            # slant left connections at top
            if i == 0:
                W[i, N1*N2-1] = 1.
            else:
                W[i, N1*N2 - N2 + i -1] = 1.

    # if periodic:
    #     c0 = np.zeros(N1*N2)
    #     c0[[1, N2-1, N2, N1*N2-N2, N1*N2-N2+1, N1*N2 - 1]] = 1.
    #     c1 = np.zeros(N1*N2)
    #     c1[[1, N2, N2+1, N1*N2-N2-1, N1*N2-N2, N1*N2-1]] = 1.

    #     even_block = scipy.linalg.circulant(c0).T
    #     odd_block = scipy.linalg.circulant(c1).T

    #     for i in range(N1):
    #         idx = slice(i*N1,(i+1)*N1)
    #         if i % 2 == 0:
    #             W[idx, :] = even_block[idx, :]
    #         else:
    #             W[idx, :] = odd_block[idx]

    W = W + W.T
    W[W > 0] = 1.

    # if periodic:
    #     assert np.all(np.sum(W, 0) == 6)

    x, y = generate_grid((N1, N2), "offset")
    coords = np.stack([x.flatten(), y.flatten()]).T
    return pygsp.graphs.Graph(W=W, coords=coords)

N1, N2 = 12, 12
tg = TriangleGrid2d(N1, N2, periodic=True)
tg.plot_signal(np.ones(N1*N2))
fig, ax = plt.subplots()
ax.matshow(tg.W.toarray())
fig.suptitle("Adjacency matrix for 6x6 hex grid\n")
fig.tight_layout()
# block circulant https://math.stackexchange.com/questions/4022364/eigenvalues-of-a-particular-block-circulant-matrix

tg.compute_fourier_basis()

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axs.flat):
    hexshow(HexArray(tg.U[:, i+1].reshape(N1, N2)), ax=ax)
fig.suptitle("12x12 Triange Graph: Eigenvector modes (with periodic boundary conds)\n")
fig.tight_layout()

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        H = HexArray(np.zeros((N1, N2)))
        H[i, j] = 1.
        hexshow(np.real(ifft(H)), ax=axs[i, j])
fig.suptitle("12x12 Triangle Graph: Hexagonal Fourier modes\n")
fig.tight_layout()


eigs = np.linalg.eigvals(tg.L.toarray())
fig, ax = plt.subplots()
ax.plot(np.sort(eigs)[::-1])
ax.set_ylabel("magnitude")
ax.set_xlabel("eigenvalue idx (sorted)")
fig.suptitle("Graph Laplacian eigenvalues: 12x12 Triangle Graph")
