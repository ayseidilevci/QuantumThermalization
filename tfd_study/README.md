# tfd_study: variational TFD / Gibbs states for 2+2, 3+3 and 4+4 qubits on Tuna-17

This package generalises the 2+2 experiment of Sagastizabal et al. (npj Quantum Inf. 7, 130) to n+n qubits and p ansatz layers. It studies the generalisation with qiskit simulators (noiseless, and noisy with a Tuna-17 model fitted to our own hardware data) and plans the next Tuna-17 run from those results.

## Notebooks (read in order)

| Notebook | Question |
|---|---|
| `notebooks/01_noiseless_design.ipynb` | Which loss exponent ς and how many layers p, without noise? How do the circuits fit on Tuna-17? |
| `notebooks/02_noise_model.ipynb` | How is Tuna-17's noise modelled when QI publishes no error rates? What does the noise cost by size and depth? |
| `notebooks/03_spsa_study.ipynb` | How close does SPSA get with a hardware-sized budget, and how robust is that? |
| `notebooks/04_hardware_plan.ipynb` | The chosen hardware experiment, its budget, and its compiled circuits. |

The notebooks only load and plot results. The computations are scripts that write to `results/`:

| Script | Writes | Run time (20 cores) |
|---|---|---|
| `python -m tfd_study.fit_noise` | `noise_fit.json` | ~10 min |
| `python -m tfd_study.run_noiseless_scan` (+ `--redo-F 10`) | `results/noiseless_scan.json` | ~35 min |
| `python -m tfd_study.run_ceilings` | `results/ceilings.json` | ~6 min |
| `python -m tfd_study.run_spsa_study` | `results/spsa_study.jsonl` (resumable) | ~1–2 h |

## Modules

- `model.py`: the transverse-field Ising model, Gibbs state, ansatz for any n and p, loss, tomography (ρ_A and ρ_B from one set of circuits), readout correction, and exact references.
- `devices.py`: Tuna-17 connectivity (`tuna17.json`), ladder placements on the chip, compilation to native gates, and the noisy simulator.
- `protocol.py`: the SPSA warm-start chain, using the SPSA that ran on Tuna-17 (`gibbs_tfd_reproduction/optimizers.py`), with a random-kick start and per-submission cost accounting.
- `run_hardware.py`: the hardware runner for any n and p. It uses the scheduler of `gibbs_tfd_reproduction/run_hardware_parallel.py` unchanged. Supports `--self-test`, `--dry-run` and `--resume`.

## Conventions

- Qubits: A1..An = q0..q_{n-1}, B1..Bn = q_n..q_{2n-1}.
- Fidelity: qiskit's `state_fidelity`, F = (Tr√…)². The paper and the tfd_qi note quote √F.
- Every study process limits numpy's BLAS/OpenMP to one thread (`tfd_study/__init__.py`). Without that, each Python process on this machine commits about 4 GB, and parallel runs die without an error.
