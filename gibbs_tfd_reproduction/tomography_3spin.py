import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


PHASE_CMAP = LinearSegmentedColormap.from_list(
    "density_matrix_phase",
    [
        (0.00, "#e52420"),
        (0.25, "#a8c83c"),
        (0.50, "#3549a5"),
        (0.75, "#55c9df"),
        (1.00, "#e52420"),
    ],
    N=256,
)

PHASE_NORM = Normalize(vmin=-np.pi, vmax=np.pi)


def plot_density_matrix_3d(ax, rho, title):
    """Plot density-matrix magnitudes as bars colored by complex phase."""

    rho = np.asarray(rho, dtype=complex)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square density matrix.")

    dimension = rho.shape[0]
    number_of_qubits = int(np.log2(dimension))

    if 2**number_of_qubits != dimension:
        raise ValueError("Density-matrix dimension must be a power of 2.")

    x, y = np.meshgrid(
        np.arange(dimension),
        np.arange(dimension),
    )

    x = x.ravel()
    y = y.ravel()

    magnitudes = np.abs(rho).ravel()
    phases = np.angle(rho).ravel()

    colors = PHASE_CMAP(PHASE_NORM(phases))

    # Lighten the colors without changing the phase-to-color mapping.
    colors[:, :3] = 0.08 + 0.92 * colors[:, :3]

    ax.bar3d(
        x - 0.4,
        y - 0.4,
        np.zeros_like(magnitudes),
        0.8,
        0.8,
        magnitudes,
        color=colors,
        edgecolor=(0.2, 0.2, 0.2, 0.5),
        linewidth=0.25,
        shade=False,
    )

    basis_labels = [
        format(i, f"0{number_of_qubits}b")
        for i in range(dimension)
    ]

    ax.set_xticks(np.arange(dimension))
    ax.set_yticks(np.arange(dimension))
    ax.set_xticklabels(basis_labels, rotation=45, ha="right")
    ax.set_yticklabels(basis_labels)

    ax.set_xlabel("Column basis state")
    ax.set_ylabel("Row basis state")
    ax.set_zlabel(r"$|\rho_{ij}|$")
    ax.set_title(title, pad=12)

    ax.set_zlim(0, max(0.01, magnitudes.max() * 1.1))
    ax.view_init(elev=25, azim=-55)


def add_phase_colorbar(fig, axes):
    """Add a shared phase colorbar."""

    mappable = ScalarMappable(
        norm=PHASE_NORM,
        cmap=PHASE_CMAP,
    )
    mappable.set_array([])

    # Place the colorbar in its own axes well to the right of both 3D plots.
    # This avoids it being squeezed between or overlapping the subplots.
    colorbar_ax = fig.add_axes([0.925, 0.22, 0.018, 0.58])
    colorbar = fig.colorbar(mappable, cax=colorbar_ax, aspect=25)

    colorbar.set_label("Phase of density-matrix element")
    colorbar.set_ticks([
        -np.pi,
        -np.pi / 2,
        0,
        np.pi / 2,
        np.pi,
    ])
    colorbar.set_ticklabels([
        r"$-\pi$",
        r"$-\pi/2$",
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
    ])

    return colorbar


def plot_gibbs_tomography_comparison(
    rho_gibbs,
    rho_reconstructed,
    beta,
    sigma,
    output_path=None,
    show=True,
):
    """
    Compare the exact Gibbs state with the reconstructed state
    for one user-selected sigma.

    The title reports the overall reduced-state fidelity.
    """

    rho_gibbs = np.asarray(rho_gibbs, dtype=complex)
    rho_reconstructed = np.asarray(rho_reconstructed, dtype=complex)

    if rho_gibbs.shape != rho_reconstructed.shape:
        raise ValueError(
            "The exact and reconstructed density matrices "
            "must have the same shape."
        )

    # Uhlmann fidelity:
    # F(rho, sigma) = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2.
    eigenvalues, eigenvectors = np.linalg.eigh(rho_gibbs)
    sqrt_rho = (
        eigenvectors
        @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
        @ eigenvectors.conj().T
    )

    product = sqrt_rho @ rho_reconstructed @ sqrt_rho
    product = (product + product.conj().T) / 2

    product_eigenvalues = np.linalg.eigvalsh(product)

    fidelity = float(
        np.sum(np.sqrt(np.clip(product_eigenvalues, 0.0, None))) ** 2
    )
    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    fig = plt.figure(figsize=(16, 7.2))

    ax_exact = fig.add_subplot(1, 2, 1, projection="3d")
    ax_reconstructed = fig.add_subplot(1, 2, 2, projection="3d")

    # First panel: exact Gibbs state.
    plot_density_matrix_3d(
        ax_exact,
        rho_gibbs,
        r"Exact Gibbs state $\rho_\beta$",
    )

    # Second panel: the chosen sigma only.
    plot_density_matrix_3d(
        ax_reconstructed,
        rho_reconstructed,
        rf"Reconstructed $\hat{{\rho}}_A$, $\varsigma={sigma:g}$",
    )

    fig.suptitle(
        rf"Gibbs-state tomography: $\beta={beta:g}$, "
        rf"$\varsigma={sigma:g}$, "
        rf"$F(\rho_\beta,\hat{{\rho}}_A)={fidelity:.6f}$",
        fontsize=15,
    )

    add_phase_colorbar(
        fig,
        [ax_exact, ax_reconstructed],
    )

    # Reserve a dedicated right margin for the phase colorbar.
    fig.subplots_adjust(
        left=0.02,
        right=0.88,
        bottom=0.06,
        top=0.82,
        wspace=0.08,
    )

    if output_path is not None:
        output_path = str(output_path)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved tomography comparison to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)

    return fidelity

