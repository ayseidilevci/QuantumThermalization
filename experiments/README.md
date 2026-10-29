# Experiment data

Hardware runs on Quantum Inspire's Tuna-17 (and the simulator runs they are compared with). Every run
directory holds `state.json` (configuration, optimizer state, tomography results) and `raw_jobs.jsonl` (the raw
counts of every job submitted). Fidelities in `state.json` are qiskit's F; the report uses the paper's
convention, √F.

## 2 spins (qubits 2, 4, 5, 8 unless stated)

Runner: `gibbs_tfd_reproduction/run_hardware_parallel.py`.

| Folder | Runs | Plots |
|---|---|---|
| `2spin_reproduction_chain` | `parallel_20261019_175824`: chain β ≈ 0 → 0.5 → 1 → 5, 4 independent runs, 20 SPSA steps per β | `reproduction_vs_paper*.png`; `manhattan_2p2*.png` inside the run (`plot_vs_paper.py`, `plot_manhattan_2p2.py`) |
| `2spin_intermediate_chain` | `parallel_20261022_140225`: β = 2, 3, 4, 5, continued from the chain at β = 1 | `intermediate_steps_vs_paper.png` |
| `2spin_cold_start` | `parallel_20261022_215422`: cold start at β = 5 (2 runs, 20 steps). +20 steps at β = 5: `_20261023_114448` (cold start), `_154440` (reproduction chain), `_114417` (intermediate chain) | `chain_vs_paper*.png`, `chains_and_cold_start_vs_paper.png`, `loss_*.png` (`plot_vs_paper.py`, `plot_cold_start_loss.py`) |
| `2spin_large_beta` | `parallel_20261023_181044`, `_181058`, `_181113`, `_181128`: cold starts at β = 10, 20, 50, 100 on qubits 12, 14, 15, 16 | `large_beta_*.png` (`plot_large_beta.py`) |
| `job_statistics` | Jobs, circuits, shots, device and queue time per experiment, from the Quantum Inspire API | `job_statistics.json` (`job_statistics.py`) |

## 3+3 and 4+4

Runners: `tfd_study/run_hardware.py` (SPSA on the device) and `tfd_study/run_mitigated.py` (fixed noiseless-optimal
angles with error mitigation). Plans are the JSON files in `tfd_study/`.

| Folder | Runs |
|---|---|
| `3p3_spsa_old_qubits` | `tfd_n3p1_20261021_235101`: SPSA, qubits 2-4-7 / 5-8-11, chain β = 0.5 → 1 → 2 → 5, 2 independent runs, 40 steps per β (`hardware_plan.json`). Same plan on the ideal simulator and the QX emulator. |
| `4p4_spsa_old_qubits` | `tfd_n4p1_20261022_053620`: SPSA, qubits 0-2-5-9 / 1-4-8-12, 1 run (`hardware_plan_4.json`); `fidelity_paper_convention.png` inside (`plot_4p4_hardware.py`). Same plan on the ideal simulator (3 runs) and the QX emulator. |
| `3p3_mitigated` | Noiseless-optimal angles at β = 1 unless stated. `mitig-hw_n3_20261027_112356`: qubit scan, 3 of 8 placements. `_141000`: step-by-step test on 2-5-9 / 4-8-12 (platform, native, twirl). `_151540`: the same test on 8-12-15 / 11-14-16. `_200737`: final run on 8-12-15 / 11-14-16, twirl, β = 0.5, 1, 2, 5, 2 repeats. `_20261028_003204`: native at β = 5, for the twirling gain. |
| `4p4_mitigated` | `mitig-hw_n4_20261028_054813`: qubits 4-8-12-15 / 7-11-14-16, native, β = 0.5, 1, 2, 5, 3 repeats. |
| `3p3_spsa_user_qubits` | `tfd_n3p1_20261028_084445`: SPSA as in `hardware_plan.json`, on 8-12-15 / 11-14-16 (`hardware_plan_3_userqubits.json`). |
| `4p4_spsa_user_qubits` | `tfd_n4p1_20261028_104418`: SPSA as in `hardware_plan_4.json`, on 4-8-12-15 / 7-11-14-16 (`hardware_plan_4_userqubits.json`). |
| `3p3_4p4_user_qubits_comparison` | Plots comparing the runs on the user's qubits: `ideal_classical_spsa.png`, fidelity versus β for the ideal simulation, the noiseless-optimal angles on Tuna-17 (`3p3_mitigated/..._200737`, `4p4_mitigated`) and SPSA on Tuna-17 (`*_spsa_user_qubits`) (`plot_ideal_classical_spsa.py`); `loss_hw_3p3_4p4.png`, the loss along the SPSA runs per step, with the same angles evaluated without noise and the noiseless-optimal angles' loss measured on the device (`plot_loss_hw.py`). |


In the mitigated runs every tomography has its own readout calibration, and `state.json` stores each result with
no, per-qubit and full readout correction; `compiled.jsonl` holds the circuits as submitted.
