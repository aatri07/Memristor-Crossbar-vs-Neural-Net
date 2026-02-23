import numpy as np

"""
Simulation of a memristor crossbar array.

The crossbar performs analog matrix-vector multiplication using Ohm's Law and Kirchhoff's 
Current Law:

    I = G @ V

Where:
    V : input voltage vector
    G : conductance matrix (memristor states)
    I : output current vector

Each output current is the sum of conductance-weighted input voltages, which is mathematically
equivalent to a neural network layer.

(Note: matrix-vector multiplication is a series of dot products)

See the README.md for more details on the memristor crossbar model and its applications in 
neuromorphic computing.
"""

class MemristorCrossbar:
    # conductance values are typically in the microSiemens range for memristors, so we initialize 
    # with small random values.
    randomConductanceFloor = 1e-6
    randomConductanceCeiling = 1e-3
    noiseStdDev = 1e-5
    # flexible poly (vinyl alcohol)-graphene oxide (PVA-GO) memristors have a threshold voltage around 0.2V, 
    # so we can use this as a reference for when to modify conductance in our simplified model.
    # source: https://www.researchgate.net/publication/340227594_Flexible_PolyVinyl_Alcohol-Graphene_Oxide_Hybrid_Nanocomposite_Based_Cognitive_Memristor_with_Pavlovian-Conditioned_Reflex_Activities#:~:text=ultralow%20voltages%20below%200.5%20V%20and%20high,power%20consumption%20during%20the%20SET%20process%20and
    voltageThreshold = 0.2

    # initialize the crossbar with given dimensions and optional initial conductance matrix.
    # @param rows: number of rows in the crossbar (output dimension)
    # @param cols: number of columns in the crossbar (input dimension)
    # @param G_init: optional initial conductance matrix (shape: rows x cols)
    def __init__(self, rows, cols, G_init=None):
        self.rows = rows
        self.cols = cols
        self.G = G_init 
        
        # if no initial conductance matrix is provided, initialize with random values in the specified range.
        if self.G is None:
            self.G = np.random.uniform(
                low=self.randomConductanceFloor,
                high=self.randomConductanceCeiling,
                size=(rows, cols)
            )
        else :
            self.G = G_init

    # perform matrix-vector multiplication to compute output currents (linear)
    # @param V: input voltage vector (shape: cols x 1)
    # @return: output current vector (shape: rows x 1)
    def compute_output(self, V):
        # ensure V is a column vector
        V = np.atleast_2d(V).reshape(-1, 1)  
        if V.shape[0] != self.cols:
            raise ValueError(f"Input voltage vector length {V.shape[0]} does not match number of columns {self.cols}")
        # Ohm's Law and Kirchhoff's Current Law: I = G @ V
        I = self.G @ V  
        return I.flatten()  # return as 1D array for convenience
    

    # apply voltage over the threshold to update memristor states (conductance)
    # @param V: input voltage vector (shape: cols x 1)
    # @return: output current vector after applying voltage and updating conductance (shape: rows x 1)
    def write (self, V):
        # ensure V is a column vector
        V = np.atleast_2d(V).reshape(-1, 1) 
        if V.shape[0] != self.cols:
            raise ValueError(f"Input size {V.shape[0]} does not match crossbar cols {self.cols}")
        
        for col in range(self.cols):
            if V[col, 0] > self.voltageThreshold:
                # larger voltage -> bigger conductance change
                deltaG = 1e-5 * (V[col, 0] - self.voltageThreshold)  
                self.modify_conductance(row=slice(None), col=col, deltaG=deltaG)
            
        # still compute output currents if desired
        I = self.G @ V
        return I.flatten()
    
    # modify the conductance of a specific memristor when voltage is high enough to cause a state change (simplified model)
    # @param row: row index of the memristor to modify
    # @param col: column index of the memristor to modify
    # @param deltaG: change in conductance (positive for increase, negative for decrease
    def modify_conductance(self, row, col, deltaG):
        # support slices or ints
        self.G[row, col] += deltaG
        # clip to avoid negative conductance
        if isinstance(row, int) and isinstance(col, int):
            self.G[row, col] = max(self.G[row, col], 0)
        else:
            self.G[row, col] = np.clip(self.G[row, col], 0, None)
    
    # apply a voltage vector and automatically decide whether to modify conductance based on the voltage threshold (simplified learning rule)
    # if all voltages < voltage threshold, only compute
    # if any voltage >= voltage threshold, write and compute
    # @param V: input voltage vector (shape: cols x 1)
    # @return: output current vector after applying voltage and potentially updating conductance (shape: rows x 1)
    def apply_voltage(self, V):
        V = np.atleast_2d(V).reshape(-1, 1)  # ensure V is a column vector
        if V.shape[0] != self.cols:
            raise ValueError(f"Input voltage vector length {V.shape[0]} does not match number of columns {self.cols}")
        if np.any(V > self.voltageThreshold):
            # voltage high enough to modify conductance (updates G and returns I)
            return self.write(V)  
        else:
            # safe to compute only (doesn't modify G, just returns I)
            return self.compute_output(V)  


    # set conductance matrix manually (for testing or specific configurations)
    # @param G_new: new conductance matrix (shape: rows x cols)
    def set_conductance(self, G_new):
        G_new = np.asarray(G_new)
        if G_new.shape != (self.rows, self.cols):
            raise ValueError(
                f"Conductance matrix shape {G_new.shape} does not match crossbar dimensions ({self.rows}, {self.cols})"
            )
        self.G = G_new
    
    # get the current conductance matrix (for inspection or analysis)
    def get_conductance(self):
        return self.G
    
    # simulate device variability by adding random noise to the conductance values
    # @param std: standard deviation of the noise (if None, uses default noiseStdDev)
    def add_noise(self, std=None):
        if std is None:
            std = self.noiseStdDev
        noise = np.random.normal(loc=0.0, scale=std, size=self.G.shape)
        self.G += noise
        # ensure conductance values remain non-negative
        self.G = np.clip(self.G, a_min=0, a_max=None)
    
    # repr for easy visualization of the crossbar state/debugging
    def __repr__(self):
        return f"MemristorCrossbar({self.rows}x{self.cols}, G=\n{self.G})"