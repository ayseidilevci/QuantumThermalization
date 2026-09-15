# H_A = ZZ_A + g·X_A as a 4×4 matrix (g=1, n=2 spins)
# Diagonalize -- eigenstates |j⟩ and eigenenergies E_j
# ρ_Gibbs(β) = Σ_j exp(-βE_j)|j⟩⟨j| / Z
# |TFD(β)⟩ = Σ_j exp(-βE_j/2)|j⟩_B|j⟩_A / √Z
# partial-trace |TFD(β)⟩⟨TFD(β)| over system B == ρ_Gibbs(β)

import numpy as np

class GibbsState:
    def __init__(self, H):
        self.H = H

    def gibbs_density_matrix(self, beta):
        Z = 0
        eigen_values, eigen_vectors = self.H
        eigen_values = eigen_values - eigen_values.min()
        sigma = np.zeros((4, 4), dtype=complex)
        for j in range(len(eigen_values)):
            vector = eigen_vectors[:, j]
            weight = np.exp(-beta*eigen_values[j])
            sigma +=  weight * np.outer(vector, vector.conj())
            Z += weight
        p = sigma / Z
        return p

class TFDState:
    def __init__(self, H_A, H_B):
        self.H_A = H_A
        self.H_B = H_B

    def tfd_state_vector(self, beta):
        Z = 0
        eigen_values_A, eigen_vectors_A = self.H_A
        eigen_values_B, eigen_vectors_B = self.H_B
        eigen_values_A = eigen_values_A - eigen_values_A.min()
        eigen_values_B = eigen_values_B - eigen_values_B.min()

        sigma = np.zeros((16), dtype=complex)
        for j in range(len(eigen_values_A)):
            vector_A = eigen_vectors_A[:, j]
            vector_B = eigen_vectors_B[:, j]
            weight = np.exp(-beta*eigen_values_A[j]/2)
            sigma += weight * np.kron(vector_B, vector_A)
            Z += weight**2
        state_tfd = sigma / np.sqrt(Z)
        return state_tfd


