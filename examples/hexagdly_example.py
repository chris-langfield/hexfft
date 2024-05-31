import hexagdly
from example_utils import *
import numpy as np
from hexfft.plot import hexshow
from hexfft import HexArray
from hexfft import fft, ifft
from hexfft.utils import filter_shift

# Tensor parameters:
num_rows = 20
num_columns = 24

def toy_data_hexarray(*args, **kwargs):
    x = toy_data(*args, **kwargs)
    return HexArray(np.squeeze(np.array(x.to_torch_tensor().T)))

# Shapes in tensor with position (px, py)
t1 = toy_data_hexarray('double_hex', num_rows, num_columns, px=5, py=5)
t2 = toy_data_hexarray('double_hex', num_rows, num_columns, px=14, py=8)
t3 = toy_data_hexarray('snowflake_3', num_rows, num_columns, px=5, py=16)
t4 = toy_data_hexarray('snowflake_3', num_rows, num_columns, px=14, py=19)

h = t1 + t2 + t3 + t4 

hexshow(h, cmap="gray_r")

kernel = HexArray(np.zeros((num_rows, num_columns)))
c1, c2 = num_rows//2, num_columns//2
kernel[c1, c2] = 1.
idx = np.array([[c1, c2-1], [c1, c2+1], [c1-1, c2], [c1+1, c2], [c1-1, c2-1], [c1+1, c2-1]])
kernel[tuple(idx.T)] = 1.
hexshow(kernel, cmap="gray_r")

H = fft(h)
hexshow(np.abs(h), cmap="gray_r")

K = fft(filter_shift(kernel))
hexshow(np.real(K), cmap="gray_r")

CONV = H * K
hexfft_conv = ifft(CONV)


##### ---------------------------

# Shapes in tensor with position (px, py)
s1 = toy_data('double_hex', num_rows, num_columns, px=5, py=5)
t1 = s1.to_torch_tensor()
s2 = toy_data('double_hex', num_rows, num_columns, px=14, py=8)
t2 = s2.to_torch_tensor()

s3 = toy_data('snowflake_3', num_rows, num_columns, px=5, py=16)
t3 = s3.to_torch_tensor()
s4 = toy_data('snowflake_3', num_rows, num_columns, px=14, py=19)
t4 = s4.to_torch_tensor()

tensor = t1 + t2 + t3 + t4 
hex_conv = hexagdly.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 1, stride = 1, bias=False, debug=True)
hex_conved_tensor = hex_conv(tensor)
hg_conv = HexArray(np.squeeze(hex_conved_tensor.detach().numpy().T))

fig, ax = plt.subplots(3, 1, figsize=(4, 12))
im = hexshow(np.real(hexfft_conv), cmap="gray_r", ax=ax[0])
fig.colorbar(im, ax=ax[0])
ax[0].set_title("hexfft results")

im = hexshow(hg_conv, cmap="gray_r", ax=ax[1])
fig.colorbar(im, ax=ax[1])
ax[1].set_title("hexagDLy results")

im = hexshow(np.real(hexfft_conv - hg_conv), cmap="gray_r", ax=ax[2])
fig.colorbar(im, ax=ax[2])
ax[2].set_title("Difference")

fig.tight_layout()

