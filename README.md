![Screenshot 2024-05-21 at 1 00 10 PM](https://github.com/chris-langfield/hexfft/assets/34426450/b4eff5e9-2375-4d2c-a77e-5009efb34495)

# hexfft

A Python package aiming to provide an easy and efficient interface to various implementations of the [Hexagonal Fast Fourier Transform](https://en.wikipedia.org/wiki/Hexagonal_fast_Fourier_transform).

## Get started

#### Plot hexagonally sampled 2D signals
```python
from hexfft import HexArray
from hexfft.plot import hexshow
import numpy as np

data = np.random.normal(size=(8, 6))
h = HexArray(data)
hexshow(h)
```
![Screenshot 2024-05-21 at 1 06 04 PM](https://github.com/chris-langfield/hexfft/assets/34426450/92d11a97-8b64-4d3f-9ac9-c612aa4b5437)

#### Perform FFT for rectangularly or hexagonally periodic signals

```python
from hexfft import fft, ifft

X = fft(h)
X_hx = fft(h, periodicity="hex")
```

#### Operate on a 3D stack

```python
from hexfft import FFT

shape = (32, 32)

fftobj = FFT(shape, periodicity="hex") # or "rect"
X = fftobj.forward(x)
xx = fftobj.inverse(X)

...
```

## Install

```
pip install hexfft
```

The only dependencies are `numpy`, `scipy`, and `matplotlib`. `pytest` is required to run the tests.

#### Developer install

```
git clone git@github.com:chris-langfield/hexfft.git
pip install -e hexfft/
cd hexfft
pytest tests/
```

## Further reading
---------------------------------------
> R. M. Mersereau, "The processing of hexagonally sampled two-dimensional signals," in Proceedings of the IEEE, vol. 67, no. 6, pp. 930-949, June 1979, doi: 10.1109/PROC.1979.11356

> Ehrhardt, J. C. (1993). Hexagonal fast Fourier transform with rectangular output. In IEEE Transactions on Signal Processing (Vol. 41, Issue 3, pp. 1469–1472). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/78.205759 
