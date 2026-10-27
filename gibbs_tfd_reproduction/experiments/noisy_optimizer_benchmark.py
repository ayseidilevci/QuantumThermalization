"""Worker code for noisy_optimizer_comparison.ipynb.

Lives in a module (not in the notebook) so concurrent.futures can run one
warm-start chain per process: on Windows, worker processes are spawned and
can only call functions they can import. Each noisy cost evaluation is a
few seconds of single-threaded Aer shot simulation, so running the
(optimizer, repeat) chains side by side is where the speed-up comes from.

One "chain" is the canonical experiment.py protocol under full Starmon-7
noise: start at the known beta -> 0 solution, optimize at each beta in turn,
and warm-start every step from the previous step's angles. After each step,
rho_A is reconstructed by state tomography (readout-corrected) and compared
with the exact Gibbs state.
"""

import sys
import time
import warnings
from pathlib import Path

# experiments/ -> gibbs_tfd_reproduction/, where the pipeline's modules live
# (they use bare imports like `from tfd_circuit import ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qi_utilities.device_simulation.simulators import NoisySimulator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Pauli, partial_trace, state_fidelity
from scipy.optimize import minimize

from cost_function import CostFunction
from ground_truth import GibbsState, default_subsystem_hamiltonian
from tfd_circuit import TFDCircuit
from tomography import extract_ro_matrices, reconstruct_density_matrix, run_tomography

# transpile() warns on every call that coupling_map/basis_gates are passed
# alongside backend (NoisySimulator does this internally) -- harmless noise.
warnings.filterwarnings("ignore", message="Providing `coupling_map`")

VARSIGMA = 1.57
INITIAL_LAYOUT = [0, 2, 3, 5]  # q0=A2, q1=A1, q2=B2, q3=B1 -> Starmon-7 qubits
OPTIMIZERS = ("COBYLA", "Nelder-Mead", "Powell", "SPSA")


def make_backend(noisy=True):
    """Starmon-7 simulator with every noise channel on (T1/T2 during delays,
    single-qubit and CZ depolarization, readout T1/T2 and assignment error).
    One Aer thread: Aer doesn't speed up these per-shot noisy runs with more
    threads, and the parallelism is across processes instead."""
    backend = NoisySimulator("Starmon-7", ideal_simulation=not noisy)
    backend.set_options(max_parallel_threads=1)
    return backend


def gibbs_target(beta):
    return GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(beta)


def fidelity(rho, beta):
    # Linear-inversion tomography can come out slightly non-physical under
    # shot noise, hence validate=False.
    return float(state_fidelity(DensityMatrix(rho), DensityMatrix(gibbs_target(beta)),
                                validate=False))


# ---------------------------------------------------------------- references

def exact_cost(params, beta, varsigma=VARSIGMA):
    """Noiseless statevector version of CostFunction's cost (same Pauli terms)."""
    sv = TFDCircuit(*params).get_state_vector()
    e = lambda p: sv.expectation_value(Pauli(p)).real
    return (e("IIXI") + e("IIIX") + e("XIII") + e("IXII")
            + varsigma * (e("IIZZ") + e("ZZII"))
            - beta ** (-varsigma) * (e("XIXI") + e("IXIX") + e("ZIZI") + e("IZIZ")))


def exact_rho_A(params):
    return partial_trace(TFDCircuit(*params).get_state_vector(), [2, 3]).data


def noiseless_optimum(beta, n_starts=30, seed=0):
    """Best angles under the exact cost (multi-start COBYLA): what a perfect
    optimizer could reach if there were no noise and no shot sampling."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(exact_cost, rng.uniform(-np.pi, np.pi, 4), args=(beta,),
                     method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def measure_ro_matrices(backend, shots=8192):
    """Readout calibration on the four layout qubits: prepare all-|0>, then
    all-|1>, same blocks CostFunction appends to its cost circuit."""
    qc = QuantumCircuit(4, 8)
    for q in range(4):
        qc.measure(q, q)
    for q in range(4):
        qc.reset(q)
    qc.x(range(4))
    for q in range(4):
        qc.measure(q, 4 + q)
    qc_t = transpile(qc, backend, initial_layout=INITIAL_LAYOUT)
    counts = backend.run(qc_t, shots=shots).result().get_counts()
    # 8-bit strings: bits 0-3 (rightmost) = prep |0>, bits 4-7 = prep |1>
    prep0, prep1 = {}, {}
    for bits, n in counts.items():
        prep0[bits[4:]] = prep0.get(bits[4:], 0) + n
        prep1[bits[:4]] = prep1.get(bits[:4], 0) + n
    return extract_ro_matrices(prep0, prep1)


def noisy_tomography(backend, params, ro_matrices, shots=8192):
    corrected, settings = run_tomography(backend, INITIAL_LAYOUT, *params,
                                         subsystem_qubits=(0, 1), shots=shots,
                                         ro_matrices=ro_matrices)
    return reconstruct_density_matrix(corrected, settings, subsystem_qubits=(0, 1))


def reference_point(beta, tomo_shots=8192):
    """For one beta: the noiseless-optimal angles, their exact rho_A, and the
    same angles run through noisy tomography. That last one is roughly the
    best this hardware can show for this beta, whatever the optimizer."""
    backend = make_backend(noisy=True)
    params = noiseless_optimum(beta)
    rho_exact = exact_rho_A(params)
    rho_noisy = noisy_tomography(backend, params, measure_ro_matrices(backend), tomo_shots)
    return {
        "beta": beta,
        "params": params,
        "rho_exact": rho_exact,
        "rho_noisy": rho_noisy,
        "fidelity_exact": fidelity(rho_exact, beta),
        "fidelity_noisy": fidelity(rho_noisy, beta),
    }


# ---------------------------------------------------------------- optimizers

class _BudgetExhausted(Exception):
    pass


class _BudgetedCost:
    """Wraps the cost so every optimizer gets exactly `max_evals` circuit
    executions per beta, recording each (params, cost) pair."""

    def __init__(self, fn, beta, max_evals):
        self.fn, self.beta, self.max_evals = fn, beta, max_evals
        self.xs, self.costs = [], []

    def __call__(self, x):
        if len(self.costs) >= self.max_evals:
            raise _BudgetExhausted()
        c = float(self.fn(np.asarray(x, dtype=float), self.beta))
        self.xs.append(np.array(x, dtype=float))
        self.costs.append(c)
        return c

    def best_x(self):
        return self.xs[int(np.argmin(self.costs))]


def spsa(fun, x0, max_evals, rng, c=0.15, alpha=0.602, gamma=0.101,
         first_step=0.3, n_calibrate=4):
    """Simultaneous-perturbation stochastic approximation (Spall's standard
    gains). Two evaluations per iteration, whatever the dimension, and built
    for noisy objectives. The step size `a` is calibrated so the first
    update moves about `first_step` rad: the raw cost spans ~1e13 at
    beta -> 0 and ~10 at beta = 5, so a fixed `a` can't work for both.
    Calibration evaluations count toward the budget."""
    x = np.array(x0, dtype=float)
    dim = len(x)

    grads = []
    for _ in range(n_calibrate):
        delta = rng.choice([-1.0, 1.0], size=dim)
        grads.append(abs(fun(x + c * delta) - fun(x - c * delta)) / (2 * c))
    g_mag = float(np.mean(grads)) or 1.0

    n_iter = (max_evals - 2 * n_calibrate) // 2
    A = 0.1 * n_iter
    a = first_step * (A + 1) ** alpha / g_mag

    for k in range(n_iter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=dim)
        g = (fun(x + ck * delta) - fun(x - ck * delta)) / (2 * ck) * delta
        x = x - ak * g
    return x


def optimize(name, cost, x0, max_evals, rng):
    """Run one optimizer on a _BudgetedCost; return the final angles.
    COBYLA keeps experiment.py's settings (rhobeg=0.5). Nelder-Mead gets an
    explicit 0.5-rad initial simplex: scipy's default is 5% of x0, which is
    ~0.005 rad from the near-zero start and far too small to move under shot
    noise."""
    try:
        if name == "COBYLA":
            r = minimize(cost, x0, method="COBYLA",
                         options={"maxiter": max_evals, "rhobeg": 0.5})
            return r.x
        if name == "Nelder-Mead":
            simplex = np.vstack([x0] + [x0 + 0.5 * e for e in np.eye(len(x0))])
            r = minimize(cost, x0, method="Nelder-Mead",
                         options={"maxfev": max_evals, "initial_simplex": simplex})
            return r.x
        if name == "Powell":
            r = minimize(cost, x0, method="Powell", options={"maxfev": max_evals})
            return r.x
        if name == "SPSA":
            return spsa(cost, x0, max_evals, rng)
    except _BudgetExhausted:
        # The hard cap tripped mid-run (e.g. inside a Powell line search):
        # fall back to the lowest cost measured so far.
        return cost.best_x()
    raise ValueError(f"unknown optimizer {name!r}")


# ---------------------------------------------------------------- one chain

def run_chain(optimizer, repeat, betas, x0, max_evals, tomo_shots=8192, seed=0):
    """The canonical warm-start protocol for one optimizer, under noise."""
    rng = np.random.default_rng([seed, repeat, OPTIMIZERS.index(optimizer)])
    backend = make_backend(noisy=True)
    cf = CostFunction(varsigma=VARSIGMA, backend=backend,
                      initial_layout=INITIAL_LAYOUT, tomography=False)

    x = np.array(x0, dtype=float)
    steps = []
    for beta in betas:
        t0 = time.time()
        cost = _BudgetedCost(cf.cost_function, beta, max_evals)
        x = np.asarray(optimize(optimizer, cost, x, max_evals, rng), dtype=float)
        t_opt = time.time() - t0

        rho = noisy_tomography(backend, x, cf.ro_matrices, tomo_shots)
        steps.append({
            "beta": beta,
            "params": x.copy(),
            "costs": np.array(cost.costs),
            "lower_bound": float(CostFunction.cost_lower_bound(beta, VARSIGMA)),
            "rho_A": rho,
            "fidelity": fidelity(rho, beta),
            # same angles, no noise: separates "optimizer found bad angles"
            # from "good angles, but noise degrades the state"
            "fidelity_ideal_angles": fidelity(exact_rho_A(x), beta),
            "n_evals": len(cost.costs),
            "seconds": time.time() - t0,
            "seconds_optimizing": t_opt,
        })
    return {"optimizer": optimizer, "repeat": repeat, "steps": steps}
