"""Physics and circuits for the n+n-qubit variational TFD study.

System A and its copy B each have n qubits; qubit order is A1..An, B1..Bn
(q0..q_{n-1} = A, q_n..q_{2n-1} = B), little-endian as everywhere in qiskit.

    H_A  = sum_bonds Z_i Z_j + g sum_i X_i       open chain, g = 1
           (n = 2 keeps the single bond of Sagastizabal et al.)
    H_AB = sum_i (X_Ai X_Bi + Z_Ai Z_Bi)

Ansatz: |Phi+>^n followed by p layers of
    U = exp(-i a2 ZZ_AB/2) exp(-i a1 XX_AB/2) exp(-i g2 (ZZ_A+ZZ_B)/2) exp(-i g1 (X_A+X_B)/2)
which for n = 2, p = 1 is exactly gibbs_tfd_reproduction/tfd_circuit.py.

Loss (the engineered cost of Sagastizabal et al., Eq. S1, unscaled):
    C_s(beta) = g<X_A+X_B> + s<ZZ_A+ZZ_B> - beta^(-s) <H_AB>
At beta = 0 it is the limit -<H_AB> (same minimiser). Every term is a product of
X's or of Z's, so two measurement settings (all-Z, all-X) give the loss for any n.

Fidelity: `fidelity` is qiskit's state_fidelity, F = (Tr sqrt(sqrt(r) s sqrt(r)))^2,
the convention used across this repo. The Sagastizabal paper and the tfd_qi
note quote the square root of it.
"""

from functools import lru_cache
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, state_fidelity

G_FIELD = 1.0


# ----------------------------------------------------------------------------- model

def bonds(n):
    """Nearest-neighbour bonds of the open chain on n sites."""
    return [(0, 1)] if n == 2 else [(i, i + 1) for i in range(n - 1)]


def hamiltonian_A(n, g=G_FIELD):
    terms = [("ZZ", [i, j], 1.0) for i, j in bonds(n)] + [("X", [i], g) for i in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).to_matrix()


@lru_cache(maxsize=None)
def _eigh_A(n, g=G_FIELD):
    return np.linalg.eigh(hamiltonian_A(n, g))


def gibbs_state(n, beta, g=G_FIELD):
    """Exact Gibbs state of system A, exp(-beta H_A) / Z."""
    e, v = _eigh_A(n, g)
    w = np.exp(-beta * (e - e.min()))
    return (v * (w / w.sum())) @ v.conj().T


def fidelity(rho, beta, n, g=G_FIELD):
    """qiskit (squared) fidelity of rho_A to the exact Gibbs state."""
    return float(state_fidelity(DensityMatrix(rho), DensityMatrix(gibbs_state(n, beta, g)), validate=False))


# ----------------------------------------------------------------------------- circuits

def A(i):
    return i


def B(i, n):
    return n + i


@lru_cache(maxsize=None)
def ansatz(n, p):
    """Parametrised 2n-qubit ansatz with 4p angles [g1, g2, a1, a2] * p (unmeasured)."""
    th = ParameterVector("theta", 4 * p)
    qc = QuantumCircuit(2 * n, name=f"tfd_n{n}_p{p}")
    for i in range(n):                      # Bell pairs A_i - B_i = TFD(beta = 0)
        qc.h(B(i, n))
        qc.cx(B(i, n), A(i))
    for layer in range(p):
        g1, g2, a1, a2 = th[4 * layer:4 * layer + 4]
        qc.rx(g1, range(2 * n))
        for i, j in bonds(n):
            qc.rzz(g2, A(i), A(j))
            qc.rzz(g2, B(i, n), B(j, n))
        for i in range(n):
            qc.rxx(a1, A(i), B(i, n))
        for i in range(n):
            qc.rzz(a2, A(i), B(i, n))
    return qc


def bound_ansatz(n, p, theta):
    qc = ansatz(n, p)
    return qc.assign_parameters(dict(zip(qc.parameters, np.asarray(theta, float))))


def measured(n, p, theta, setting):
    """Ansatz + basis change + measurement of all 2n qubits (qubit k -> clbit k).

    `setting` is one letter per qubit from 'XYZ' (or 'I', measured in Z), or a
    single letter applied to every qubit."""
    if len(setting) == 1:
        setting = setting * (2 * n)
    qc = QuantumCircuit(2 * n, 2 * n)
    qc.compose(bound_ansatz(n, p, theta), inplace=True)
    for q, basis in enumerate(setting):
        if basis == "X":
            qc.ry(-np.pi / 2, q)
        elif basis == "Y":
            qc.rx(np.pi / 2, q)
    qc.measure(range(2 * n), range(2 * n))
    qc.name = f"n{n}p{p}_{setting}"
    return qc


def calibration_circuits(n):
    """Readout calibration: all qubits in |0>, all in |1> (per-qubit model)."""
    zero = QuantumCircuit(2 * n, 2 * n, name="cal0")
    zero.measure(range(2 * n), range(2 * n))
    one = QuantumCircuit(2 * n, 2 * n, name="cal1")
    one.x(range(2 * n))
    one.measure(range(2 * n), range(2 * n))
    return [zero, one]


def tomography_settings(n):
    """3^n settings; qubit A_i and B_i are measured in the same basis, so one set
    of circuits reconstructs both rho_A and rho_B."""
    return ["".join(s) * 2 for s in product("XYZ", repeat=n)]


# ----------------------------------------------------------------------------- data processing

def counts_to_probs(counts, nbits):
    """Counts -> probability vector indexed little-endian (bit k = clbit k)."""
    p = np.zeros(2 ** nbits)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] += c
    return p / p.sum()


def readout_matrices(p_cal0, p_cal1, nbits):
    """Per-qubit 2x2 confusion matrices M[q][measured, prepared] from the two calibrations."""
    mats = []
    for q in range(nbits):
        bit = (np.arange(2 ** nbits) >> q) & 1
        e01 = p_cal0[bit == 1].sum()        # prepared 0, read 1
        e10 = p_cal1[bit == 0].sum()        # prepared 1, read 0
        mats.append(np.array([[1 - e01, e10], [e01, 1 - e10]]))
    return mats


def correct_readout(p, mats):
    """Apply the inverse of the tensored confusion matrix to a probability vector."""
    nbits = len(mats)
    t = p.reshape([2] * nbits)              # axis 0 = most significant = qubit nbits-1
    for q, m in enumerate(mats):
        axis = nbits - 1 - q
        t = np.moveaxis(np.tensordot(np.linalg.inv(m), t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def parity(p, qubits):
    """<prod_{q in qubits} Z_q> from a probability vector."""
    mask = sum(1 << q for q in qubits)
    idx = np.arange(len(p))
    sign = 1 - 2 * (np.array([bin(i & mask).count("1") for i in idx]) & 1)
    return float(sign @ p)


def loss_terms(pz, px, n):
    """The six aggregate expectation values of the loss, from the all-Z and all-X distributions."""
    ZZ = sum(parity(pz, [A(i), A(j)]) + parity(pz, [B(i, n), B(j, n)]) for i, j in bonds(n))
    X = sum(parity(px, [q]) for q in range(2 * n))
    HAB = sum(parity(px, [A(i), B(i, n)]) + parity(pz, [A(i), B(i, n)]) for i in range(n))
    ZZ_A = sum(parity(pz, [A(i), A(j)]) for i, j in bonds(n))
    X_A = sum(parity(px, [A(i)]) for i in range(n))
    return {"X": X, "ZZ": ZZ, "HAB": HAB, "E_A": ZZ_A + G_FIELD * X_A}


def loss(terms, beta, s, g=G_FIELD):
    """The paper's C_s(beta); at beta = 0 the limit -<H_AB>, which has the same minimiser."""
    if beta == 0:
        return -terms["HAB"]
    return g * terms["X"] + s * terms["ZZ"] - beta ** (-s) * terms["HAB"]


def tomography_rho(probs_by_setting, n, system="A"):
    """Linear-inversion rho of system A (qubits 0..n-1) or B (n..2n-1) from the 3^n
    tomography settings, projected to a physical state."""
    off = 0 if system == "A" else n
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}
    rho = np.zeros((2 ** n, 2 ** n), complex)
    for label in product("IXYZ", repeat=n):            # label[k] acts on A_k
        support = [k for k in range(n) if label[k] != "I"]
        vals = [parity(p, [off + k for k in support]) for s, p in probs_by_setting.items()
                if all(s[off + k] == label[k] for k in support)]
        op = np.array([[1.0]])
        for k in reversed(range(n)):                   # little-endian kron: A_{n-1} leftmost
            op = np.kron(op, paulis[label[k]])
        rho += np.mean(vals) * op
    return project_physical(rho / 2 ** n)


def project_physical(rho):
    """Closest density matrix (Smolin, Gambetta, Smith 2012): clip eigenvalues, keep trace 1."""
    rho = (rho + rho.conj().T) / 2
    w, v = np.linalg.eigh(rho)
    w = w[::-1]; v = v[:, ::-1]
    acc = 0.0
    for i in range(len(w) - 1, -1, -1):
        if w[i] + acc / (i + 1) >= 0:
            w[:i + 1] += acc / (i + 1)
            break
        acc += w[i]
        w[i] = 0.0
    return (v * w) @ v.conj().T
