from models.memristor_crossbar import MemristorCrossbar
import numpy as np

# small 2x2 crossbar
# Conductance values in Siemens (S), e.g., 1e-4 S = 100 μS
# WARNING: Values above G_max (1000 μS) will saturate, values below G_min (1 μS) will be too noisy to be useful.
W = np.array([[1e-4, 2e-4],
              [3e-4, 4e-4]])
crossbar = MemristorCrossbar(rows=2, cols=2, read_noise=False)
crossbar.set_conductance(W=W)

# input voltage vector
V = np.array([1.0, 2.0])  
I = crossbar.compute_output(V)
print("Input V:", V)
print("Effective weight matrix W:\n", crossbar.get_conductance())
print("Output I:", I)
# Expected: I = W @ V = [1*1 + 2*2, 3*1 + 4*2] e-3 scaled accordingly