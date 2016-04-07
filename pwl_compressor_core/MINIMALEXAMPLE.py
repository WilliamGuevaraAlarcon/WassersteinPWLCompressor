
## Minimal example:
from pwl_compressor_core.compressor import PWLcompressor
Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
PWLapprox = PWLcompressor(Sample, Accuracy=0.01)
