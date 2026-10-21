# H_A = ZZ_A + g·X_A as a 4×4 matrix (g=1, n=2 spins)
# Diagonalize -- eigenstates |j⟩ and eigenenergies E_j
# ρ_Gibbs(β) = Σ_j exp(-βE_j)|j⟩⟨j| / Z
# |TFD(β)⟩ = Σ_j exp(-βE_j/2)|j⟩_B|j⟩_A / √Z
# partial-trace |TFD(β)⟩⟨TFD(β)| over system B == ρ_Gibbs(β)

import numpy as np

_X = np.array([[0, 1], [1, 0]])
_Z = np.array([[1, 0], [0, -1]])
_I = np.array([[1, 0], [0, 1]])


def default_subsystem_hamiltonian():
    """H_A = Z⊗Z + X⊗I + I⊗X, the 2-qubit (per-side) reference Hamiltonian
    experiment.py's reproduction script checks fidelity against — factored
    out here so the GUI can run that same ground-truth check (see
    frontend/experiment.py). Pre-diagonalized as (eigenvalues,
    eigenvectors), the form GibbsState/TFDState both expect.
    """
    H = np.kron(_Z, _Z) + np.kron(_X, _I) + np.kron(_I, _X)
    return np.linalg.eigh(H)

def default_three_spin_subsystem_hamiltonian(g=1.0, periodic=False):
    """
    3-spin transverse-field Ising Hamiltonian.

    Open chain:
        H = Z1 Z2 + Z2 Z3 + g (X1 + X2 + X3)

    If periodic=True:
        H = Z1 Z2 + Z2 Z3 + Z3 Z1
            + g (X1 + X2 + X3)
    """
    # Z1 Z2 = Z ⊗ Z ⊗ I
    ZZ_12 = np.kron(np.kron(_Z, _Z),_I)
    
    # Z2 Z3 = I ⊗ Z ⊗ Z
    ZZ_23 = np.kron(np.kron(_I, _Z),_Z)

    # X1 = X ⊗ I ⊗ I
    X_1 = np.kron(np.kron(_X, _I),_I)

    # X2 = I ⊗ X ⊗ I
    X_2 = np.kron(np.kron(_I, _X),_I)

    # X3 = I ⊗ I ⊗ X
    X_3 = np.kron(np.kron(_I, _I),_X)

    H = ZZ_12 + ZZ_23 + g * (X_1 + X_2 + X_3)

    if periodic:
        # Z3 Z1 = Z ⊗ I ⊗ Z
        ZZ_31 = np.kron(np.kron(_Z, _I),_Z)

        H += ZZ_31

    return np.linalg.eigh(H)

class GibbsState:
    def __init__(self, H):
        self.H = H

    def gibbs_density_matrix(self, beta):
        Z = 0

        eigen_values, eigen_vectors = self.H
        eigen_values = eigen_values - eigen_values.min()
        dim = len(eigen_values)
        sigma = np.zeros((dim, dim), dtype=complex)

        for j in range(len(eigen_values)):
            vector = eigen_vectors[:, j]
            weight = np.exp(-beta * eigen_values[j])

            sigma += weight * np.outer(
                vector,
                vector.conj()
            )

            Z += weight

        return sigma / Z


class TFDState:
    def __init__(self, H_A, H_B):
        self.H_A = H_A
        self.H_B = H_B

    def tfd_state_vector(self, beta):
        Z = 0
        eigen_values_A, eigen_vectors_A = self.H_A
        eigen_values_B, eigen_vectors_B = self.H_B

        # Energy shifts do not change the normalized Gibbs state
        eigen_values_A = eigen_values_A - eigen_values_A.min()
        eigen_values_B = eigen_values_B - eigen_values_B.min()

        dim_A = len(eigen_values_A)
        dim_B = len(eigen_values_B)
        if dim_A != dim_B:
            raise ValueError(
                "TFD requires A and B to have the same Hilbert-space dimension."
            )

        # A has dimension 8, B has dimension 8:
        # |TFD> lives in H_B ⊗ H_A, dimension 8*8 = 64.
        sigma = np.zeros(dim_A * dim_B, dtype=complex)

        for j in range(dim_A):
            vector_A = eigen_vectors_A[:, j]
            vector_B = eigen_vectors_B[:, j]
            weight = np.exp(-beta * eigen_values_A[j] / 2)
            sigma += weight * np.kron(vector_B,vector_A)
            Z += weight**2

        return sigma / np.sqrt(Z)
