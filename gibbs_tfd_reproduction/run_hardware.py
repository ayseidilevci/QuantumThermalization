"""Hybrid SPSA run on real Quantum Inspire hardware.

The optimizer runs on this machine; every cost evaluation is a real circuit
job on the backend (SPSA's +/- pair is batched into one job). Per beta:

  1. pre-optimize the angles on an exact noiseless simulation (seconds)
  2. SPSA gain calibration on hardware        (n_calibrate jobs)
  3. tomography of the simulator angles       (1 job)  -> "baseline" fidelity
  4. SPSA fine-tuning on hardware             (iters jobs)
  5. tomography of the SPSA angles            (1 job)  -> "final" fidelity

Comparing 3 and 5 answers whether optimizing on the device beats running
simulator-optimal angles on it.

Everything is checkpointed to <run dir>/state.json after every job, and
every job's id and raw per-block counts go to <run dir>/raw_jobs.jsonl; a crash
or Ctrl-C loses at most the job in flight. Usage (from the repo root):

  python gibbs_tfd_reproduction/run_hardware.py --dry-run          # local noisy sim, minutes
  python gibbs_tfd_reproduction/run_hardware.py --timing-test      # ONE job on Tuna-17
  python gibbs_tfd_reproduction/run_hardware.py --iters 30         # the real run
  python gibbs_tfd_reproduction/run_hardware.py --resume results/hardware_<tag>
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, Pauli, partial_trace, state_fidelity
from scipy.optimize import minimize

from cost_function import CostFunction
from ground_truth import GibbsState, default_subsystem_hamiltonian
from optimizers import spsa
from tfd_circuit import TFDCircuit
from tomography import plot_density_matrix, reconstruct_density_matrix, run_tomography

warnings.filterwarnings("ignore", message="Providing `coupling_map`")

VARSIGMA = 1.57


# ------------------------------------------------------------------ helpers

def gibbs_target(beta):
    return GibbsState(default_subsystem_hamiltonian()).gibbs_density_matrix(beta)


def fidelity(rho, beta):
    # linear-inversion tomography can be slightly non-physical -> validate=False
    return float(state_fidelity(DensityMatrix(rho), DensityMatrix(gibbs_target(beta)),
                                validate=False))


def exact_cost(params, beta):
    sv = TFDCircuit(*params).get_state_vector()
    e = lambda p: sv.expectation_value(Pauli(p)).real
    return (e("IIXI") + e("IIIX") + e("XIII") + e("IXII")
            + VARSIGMA * (e("IIZZ") + e("ZZII"))
            - beta ** (-VARSIGMA) * (e("XIXI") + e("IXIX") + e("ZIZI") + e("IZIZ")))


def simulator_optimum(beta, n_starts=20, seed=0):
    """Best angles under the exact, noiseless cost (multi-start COBYLA)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(exact_cost, rng.uniform(-np.pi, np.pi, 4), args=(beta,),
                     method="COBYLA", options={"maxiter": 1000, "rhobeg": 0.5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def with_retry(fn, what, tries=6, wait=60):
    """Run fn(); on failure wait and retry, doubling the wait up to 30 min.
    Covers the device going offline/calibrating and network hiccups."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt == tries:
                raise
            print(f"  ! {what} failed (attempt {attempt}/{tries}): {exc}\n"
                  f"    retrying in {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 1800)


def make_backend(args):
    if args.dry_run:
        from qi_utilities.device_simulation.simulators import NoisySimulator
        return NoisySimulator("Starmon-7", ideal_simulation=False)
    from qiskit_quantuminspire.qi_provider import QIProvider
    backend = QIProvider().get_backend(args.backend)
    if args.shots > backend.max_shots:
        sys.exit(f"{args.backend} allows at most {backend.max_shots} shots")
    return backend


# ------------------------------------------------------------------ state

def save_state(run_dir, state):
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(run_dir / "state.json")  # atomic: never a half-written file


def tomography_job(backend, cf, params, args, label, beta):
    corrected, settings = run_tomography(backend, cf.initial_layout, *params,
                                         subsystem_qubits=(0, 1), shots=args.shots,
                                         ro_matrices=cf.ro_matrices,
                                         raw_log_path=cf.raw_log_path,
                                         meta={"label": label, "beta": beta})
    return reconstruct_density_matrix(corrected, settings, subsystem_qubits=(0, 1))


def plot_beta(run_dir, beta, entry):
    costs = np.array(entry["spsa"]["costs"])
    lower = CostFunction.cost_lower_bound(beta, VARSIGMA)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if len(costs):
        ax.plot(costs.mean(axis=1), marker="o", ms=3, label="SPSA (mean of +/- pair)")
    ax.axhline(entry["sim_cost"], color="tab:green", ls=":", label="simulator angles (exact)")
    ax.axhline(lower, color="tab:red", ls="--", label=f"lower bound = {lower:.2f}")
    ax.set_xlabel("SPSA iteration (one hardware job each)")
    ax.set_ylabel("cost C(beta)")
    ax.set_title(f"beta={beta}: baseline fid={entry['fid_baseline']:.3f}, "
                 f"final fid={entry.get('fid_final', float('nan')):.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"cost_beta{beta}.png", dpi=120)
    plt.close(fig)

    for key in ("baseline", "final"):
        if f"rho_{key}" in entry:
            rho = np.array(entry[f"rho_{key}"][0]) + 1j * np.array(entry[f"rho_{key}"][1])
            plot_density_matrix(rho, title=f"{key}, beta={beta}",
                                save_path=str(run_dir / f"rho_{key}_beta{beta}.png"),
                                rho_target=gibbs_target(beta))


def as_json_rho(rho):
    return [np.real(rho).tolist(), np.imag(rho).tolist()]


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="Tuna-17")
    p.add_argument("--layout", type=int, nargs=4, default=[2, 4, 5, 8],
                   help="physical qubits for logical q0..q3 (A2, A1, B2, B1); must form a ring")
    p.add_argument("--protocol", choices=["finetune", "chain"], default="finetune",
                   help="finetune: each beta starts at its simulator optimum. "
                        "chain: experiment.py's protocol -- start at --x0 for the first beta, "
                        "warm-start every next beta from the previous one's result")
    p.add_argument("--x0", type=float, default=0.1, help="chain: initial value of all 4 angles")
    p.add_argument("--betas", type=float, nargs="+", default=None,
                   help="default: 0.5 1 5 (finetune), 1e-8 0.5 1 5 (chain)")
    p.add_argument("--iters", type=int, default=None,
                   help="SPSA iterations (= hardware jobs) per beta (default 30, dry run 3)")
    p.add_argument("--calibrate", type=int, default=3, help="SPSA gain-calibration jobs per beta")
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument("--first-step", type=float, default=None,
                   help="first SPSA step (and per-step cap) in rad; default 0.15 for finetune "
                        "(starts near the optimum), 0.3 for chain (must travel)")
    p.add_argument("--dry-run", action="store_true",
                   help="local noisy Starmon-7 simulator instead of hardware, tiny budget")
    p.add_argument("--timing-test", action="store_true",
                   help="submit ONE batched cost job and report how long it took")
    p.add_argument("--resume", type=Path, help="run directory of an interrupted run")
    args = p.parse_args()

    if args.resume:
        state = json.loads((args.resume / "state.json").read_text())
        saved = state["args"]
        for k in ("backend", "layout", "betas", "iters", "calibrate", "shots", "first_step", "dry_run"):
            setattr(args, k, saved[k])
        args.protocol = saved.get("protocol", "finetune")
        args.x0 = saved.get("x0", 0.1)
        run_dir = args.resume
        print(f"Resuming {run_dir}")
    else:
        chain = args.protocol == "chain"
        if args.betas is None:
            args.betas = [1e-8, 0.5, 1, 5] if chain else [0.5, 1, 5]
        if args.first_step is None:
            args.first_step = 0.3 if chain else 0.15
        if args.dry_run:
            args.backend, args.layout = "Starmon-7 (local noisy sim)", [0, 2, 3, 5]
            args.calibrate, args.shots = 1, 2048
            if args.iters is None:
                args.iters = 3
        elif args.iters is None:
            args.iters = 30
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("results") / f"{'dryrun' if args.dry_run else 'hardware'}_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                 "betas": {}}

    backend = make_backend(args)
    cf = CostFunction(varsigma=VARSIGMA, backend=backend, initial_layout=args.layout,
                      tomography=False, results_dir=str(run_dir), run_tag="evals",
                      shots=args.shots, raw_log_path=str(run_dir / "raw_jobs.jsonl"))
    if "ro_matrices" in state:
        cf.ro_matrices = {k: np.array(v) for k, v in state["ro_matrices"].items()}

    def checkpoint():
        if cf.ro_matrices is not None:
            state["ro_matrices"] = {k: v.tolist() for k, v in cf.ro_matrices.items()}
        save_state(run_dir, state)

    if args.timing_test:
        x = simulator_optimum(args.betas[0])
        t0 = time.time()
        f = cf.cost_function_batch([x + 0.05, x - 0.05], args.betas[0])
        dt = time.time() - t0
        jobs = len(args.betas) * (args.iters + args.calibrate + 2)
        print(f"\nOne batched job took {dt:.0f}s (costs {f[0]:.3f}, {f[1]:.3f}).\n"
              f"Full run = {jobs} jobs -> roughly {jobs * dt / 3600:.1f} h at this pace.")
        return

    print(f"Run directory: {run_dir}\nBackend: {args.backend}, layout {args.layout}, "
          f"betas {args.betas}, {args.iters} SPSA jobs/beta, {args.shots} shots\n"
          f"Protocol: {args.protocol}"
          + (f" (start all angles at {args.x0}, warm-start through beta)" if args.protocol == "chain" else ""))

    x_prev = np.full(4, args.x0)  # chain: where the next beta starts

    for beta in args.betas:
        key = str(beta)
        entry = state["betas"].setdefault(key, {})
        if entry.get("done"):
            print(f"\n=== beta={beta}: already done, skipping")
            x_prev = np.array(entry["x_final"])
            continue
        print(f"\n=== beta={beta}")

        if "x_sim" not in entry:
            x_sim = simulator_optimum(beta)
            entry["x_sim"] = x_sim.tolist()
            entry["sim_cost"] = float(exact_cost(x_sim, beta))
            entry["fid_sim_exact"] = fidelity(partial_trace(TFDCircuit(*x_sim).get_state_vector(), [2, 3]).data, beta)
            checkpoint()
            print(f"simulator optimum: {np.round(x_sim, 3)}  exact fidelity {entry['fid_sim_exact']:.4f}")
        x_sim = np.array(entry["x_sim"])
        if "x_start" not in entry:
            entry["x_start"] = (x_prev if args.protocol == "chain" else x_sim).tolist()
            checkpoint()
        x_start = np.array(entry["x_start"])
        if args.protocol == "chain":
            print(f"SPSA starts from {np.round(x_start, 3)} (exact fidelity "
                  f"{fidelity(partial_trace(TFDCircuit(*x_start).get_state_vector(), [2, 3]).data, beta):.4f})")

        def fun_pair(x_plus, x_minus):
            return tuple(with_retry(lambda: cf.cost_function_batch([x_plus, x_minus], beta),
                                    f"cost job (beta={beta})"))

        def on_iter(spsa_state):
            entry["spsa"] = spsa_state
            checkpoint()
            # baseline tomography right after gain calibration: by then the
            # cost jobs have measured the readout matrices it needs
            if "rho_baseline" not in entry:
                rho = with_retry(lambda: tomography_job(backend, cf, x_sim, args, "baseline", beta), "baseline tomography")
                entry["rho_baseline"] = as_json_rho(rho)
                entry["fid_baseline"] = fidelity(rho, beta)
                checkpoint()
                print(f"baseline (simulator angles on {args.backend}): fidelity {entry['fid_baseline']:.4f}")

        x_final, spsa_state = spsa(fun_pair, x_start, args.iters, first_step=args.first_step,
                                   n_calibrate=args.calibrate, seed=int(beta * 1000),
                                   state=entry.get("spsa"), on_iter=on_iter)
        entry["x_final"] = np.asarray(x_final).tolist()

        rho = with_retry(lambda: tomography_job(backend, cf, x_final, args, "final", beta), "final tomography")
        entry["rho_final"] = as_json_rho(rho)
        entry["fid_final"] = fidelity(rho, beta)
        entry["fid_final_exact"] = fidelity(
            partial_trace(TFDCircuit(*x_final).get_state_vector(), [2, 3]).data, beta)
        entry["done"] = True
        x_prev = np.asarray(x_final)
        checkpoint()
        plot_beta(run_dir, beta, entry)
        print(f"beta={beta}: baseline {entry['fid_baseline']:.4f} -> after SPSA {entry['fid_final']:.4f} "
              f"(SPSA angles without noise: {entry['fid_final_exact']:.4f})")

    print("\n beta   sim-exact  baseline(hw)  SPSA(hw)")
    for key, e in state["betas"].items():
        print(f" {key:<6} {e['fid_sim_exact']:.4f}     {e['fid_baseline']:.4f}        {e['fid_final']:.4f}")
    print(f"\nResults, plots and state in {run_dir}")


if __name__ == "__main__":
    main()
