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

Weights are represented as differential pairs of memristors (G_pos, G_neg), where the 
effective signed weight is W = G_pos - G_neg. This is necessary because memristor 
conductance is strictly non-negative, but neural network weights must be able to go negative.

Learning is performed via a delta rule: weight updates are derived from an error signal,
not raw input magnitude. This makes the crossbar's learning comparable to a single-layer
neural network trained with gradient descent.

See the README.md for more details on the memristor crossbar model and its applications in 
neuromorphic computing.
"""

class MemristorCrossbar:
    # conductance values are typically in the microSiemens range for memristors, so we initialize 
    # with small random values.
    randomConductanceFloor = 1e-6
    randomConductanceCeiling = 1e-3
    noiseStdDev = 1e-6
    # flexible poly (vinyl alcohol)-graphene oxide (PVA-GO) memristors have a threshold voltage around 0.2V, 
    # so we can use this as a reference for when to modify conductance in our simplified model.
    # source: https://www.researchgate.net/publication/340227594_Flexible_PolyVinyl_Alcohol-Graphene_Oxide_Hybrid_Nanocomposite_Based_Cognitive_Memristor_with_Pavlovian-Conditioned_Reflex_Activities#:~:text=ultralow%20voltages%20below%200.5%20V%20and%20high,power%20consumption%20during%20the%20SET%20process%20and
    voltageThreshold = 0.2
    # physical conductance bounds: G_min prevents memristors from fully switching off,
    # G_max reflects the saturation conductance of the device.
    # these are representative values for oxide-based memristors (e.g. HfOx, TaOx).
    G_min = 1e-6
    G_max = 1e-3

    # initialize the crossbar with given dimensions and optional initial conductance matrices.
    # uses differential pairs (G_pos, G_neg) so that effective weights W = G_pos - G_neg can be signed.
    # @param rows: number of rows in the crossbar (output dimension)
    # @param cols: number of columns in the crossbar (input dimension)
    # @param G_pos_init: optional initial positive conductance matrix (shape: rows x cols)
    # @param G_neg_init: optional initial negative conductance matrix (shape: rows x cols)
    # @param learning_rate: scales the magnitude of conductance updates during training
    # @param read_noise: if True, gaussian noise is injected on every forward pass to model device variability
    def __init__(self, rows, cols, G_pos_init=None, G_neg_init=None, learning_rate=1e-4, read_noise=True):
        self.rows = rows
        self.cols = cols
        self.learning_rate = learning_rate
        self.read_noise = read_noise

        # initialize positive conductance half of each differential pair
        if G_pos_init is None:
            self.G_pos = np.random.uniform(
                low=self.randomConductanceFloor,
                high=self.randomConductanceCeiling,
                size=(rows, cols)
            )
        else:
            self.G_pos = np.asarray(G_pos_init, dtype=float)

        # initialize negative conductance half of each differential pair
        if G_neg_init is None:
            self.G_neg = np.random.uniform(
                low=self.randomConductanceFloor,
                high=self.randomConductanceCeiling,
                size=(rows, cols)
            )
        else:
            self.G_neg = np.asarray(G_neg_init, dtype=float)

        # clip both halves to physical bounds on init
        self.G_pos = np.clip(self.G_pos, self.G_min, self.G_max)
        self.G_neg = np.clip(self.G_neg, self.G_min, self.G_max)

    # compute the effective signed weight matrix from the differential pair.
    # W = G_pos - G_neg, so W is in the range [G_min - G_max, G_max - G_min].
    # @return: effective weight matrix (shape: rows x cols)
    @property
    def G(self):
        return self.G_pos - self.G_neg

    # perform matrix-vector multiplication to compute output currents (linear).
    # optionally injects read noise on every call to model device variability during inference.
    # @param V: input voltage vector or batch (shape: cols x batch_size, or cols,)
    # @return: output current vector (shape: rows x batch_size)
    def compute_output(self, V):
        # ensure V is a 2D column-oriented matrix (cols x batch_size)
        V = np.atleast_2d(V)
        if V.shape[0] != self.cols:
            V = V.T
        if V.shape[0] != self.cols:
            raise ValueError(f"Input voltage shape {V.shape} does not match crossbar columns {self.cols}")

        # Ohm's Law and Kirchhoff's Current Law: I = G @ V
        # use effective signed weight matrix from differential pair
        G_eff = self.G

        # inject read noise into effective conductance if enabled, modeling per-read device variability
        if self.read_noise:
            noise = np.random.normal(loc=0.0, scale=self.noiseStdDev, size=G_eff.shape)
            G_eff = G_eff + noise

        # batch shape: (rows x batch_size)
        I = G_eff @ V
        return I

    # update conductance using the delta rule: deltaG is proportional to the outer product of
    # the error signal and the input voltage, analogous to gradient descent on a linear layer.
    # positive error drives G_pos up / G_neg down (SET/RESET); negative error does the reverse.
    # voltage magnitude must exceed voltageThreshold to physically cause a state change.
    # @param V: input voltage vector used in the forward pass (shape: cols,)
    # @param error: error signal at the output (shape: rows,), e.g. (target - actual output)
    def write(self, V, error):
        V = np.atleast_1d(V).flatten()
        error = np.atleast_1d(error).flatten()

        if V.shape[0] != self.cols:
            raise ValueError(f"Input voltage length {V.shape[0]} does not match crossbar columns {self.cols}")
        if error.shape[0] != self.rows:
            raise ValueError(f"Error signal length {error.shape[0]} does not match crossbar rows {self.rows}")

        # only apply updates at crosspoints where the input voltage exceeds the physical threshold,
        # reflecting that sub-threshold voltages cannot cause a memristive state change.
        active = np.abs(V) > self.voltageThreshold  # shape: (cols,)

        # delta rule: outer product of error and input gives a (rows x cols) update matrix,
        # scaled by learning rate. this is equivalent to the weight gradient in a linear layer.
        deltaG = self.learning_rate * np.outer(error, V)  # shape: (rows x cols)

        # zero out updates for sub-threshold inputs
        deltaG[:, ~active] = 0.0

        # apply differential update: positive delta increases G_pos / decreases G_neg (SET),
        # negative delta does the reverse (RESET). this encodes signed gradient steps
        # as two non-negative physical conductance changes.
        self.G_pos += np.maximum(deltaG, 0)
        self.G_neg -= np.minimum(deltaG, 0)  # subtracting a negative = adding a positive

        # clip both halves to physical device bounds to prevent saturation or negative conductance
        self.G_pos = np.clip(self.G_pos, self.G_min, self.G_max)
        self.G_neg = np.clip(self.G_neg, self.G_min, self.G_max)

    # apply a voltage vector and automatically decide whether to modify conductance based on the voltage threshold.
    # if no input voltage exceeds the threshold, only compute output (read operation).
    # if any input voltage exceeds the threshold, a write is possible — but a write still requires
    # an error signal, so this method is only for inference. use write() explicitly during training.
    # @param V: input voltage vector (shape: cols,)
    # @return: output current vector (shape: rows x 1)
    def apply_voltage(self, V):
        return self.compute_output(V)

    # set both conductance matrices manually (for testing or specific configurations).
    # accepts either a single signed weight matrix W (which is split into G_pos/G_neg),
    # or explicit G_pos and G_neg matrices.
    # @param W: signed weight matrix (shape: rows x cols). G_pos = max(W,0), G_neg = max(-W,0), both offset by G_min.
    # @param G_pos: optional explicit positive conductance matrix (shape: rows x cols)
    # @param G_neg: optional explicit negative conductance matrix (shape: rows x cols)
    def set_conductance(self, W=None, G_pos=None, G_neg=None):
        if W is not None:
            W = np.asarray(W, dtype=float)
            if W.shape != (self.rows, self.cols):
                raise ValueError(f"Weight matrix shape {W.shape} does not match crossbar dimensions ({self.rows}, {self.cols})")
            # decompose signed weights into differential pair, offset by G_min to keep both physical
            self.G_pos = np.clip(np.maximum(W, 0) + self.G_min, self.G_min, self.G_max)
            self.G_neg = np.clip(np.maximum(-W, 0) + self.G_min, self.G_min, self.G_max)
        else:
            if G_pos is not None:
                self.G_pos = np.clip(np.asarray(G_pos, dtype=float), self.G_min, self.G_max)
            if G_neg is not None:
                self.G_neg = np.clip(np.asarray(G_neg, dtype=float), self.G_min, self.G_max)

    # get the current effective signed weight matrix (for inspection or analysis).
    # @return: effective weight matrix W = G_pos - G_neg (shape: rows x cols)
    def get_conductance(self):
        return self.G

    # simulate device variability by adding random noise to both conductance matrices.
    # this models write noise (e.g. cycle-to-cycle variability) rather than read noise,
    # which is injected automatically in compute_output if read_noise=True.
    # @param std: standard deviation of the noise (if None, uses default noiseStdDev)
    def add_noise(self, std=None):
        if std is None:
            std = self.noiseStdDev
        noise_pos = np.random.normal(loc=0.0, scale=std, size=self.G_pos.shape)
        noise_neg = np.random.normal(loc=0.0, scale=std, size=self.G_neg.shape)
        self.G_pos = np.clip(self.G_pos + noise_pos, self.G_min, self.G_max)
        self.G_neg = np.clip(self.G_neg + noise_neg, self.G_min, self.G_max)

    # repr for easy visualization of the crossbar state/debugging
    def __repr__(self):
        return (
            f"MemristorCrossbar({self.rows}x{self.cols})\n"
            f"  G_pos=\n{self.G_pos}\n"
            f"  G_neg=\n{self.G_neg}\n"
            f"  W (effective)=\n{self.G}"
        )