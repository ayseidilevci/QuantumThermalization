# QuantumThermalization

Prepares thermal (Gibbs) states on a quantum computer through the Quantum Inspire cloud, following the variational thermofield-double protocol of Sagastizabal et al., *Variational preparation of finite-temperature states on a quantum computer*, npj Quantum Information 7, 130 (2021).

Two copies A and B of a transverse-field Ising chain are prepared in an entangled thermofield-double state, whose half A is the Gibbs state at inverse temperature β. A classical optimizer tunes the circuit's angles γ and α by minimising an engineered cost function. A desktop app (PySide6) runs the whole experiment on a local emulator or on real Quantum Inspire hardware.

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/ayseidilevci/QuantumThermalization.git
   cd QuantumThermalization
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv

   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. In VS Code, select the environment as your interpreter: `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS), then "Python: Select Interpreter".

Developed and tested with Python 3.12 and 3.13.

## Running the app

```bash
python main.py
```

Log in with your Quantum Inspire account: the app shows a verification code and opens the Quantum Inspire login page in your browser. Then:

1. **Home** explains the physics, with flash cards on the key concepts. Choose **Start**, **Past Experiments** or **Tutorial**.
2. **Backend:** pick a real device (Tuna-9, Tuna-17) or a simulator (Starmon-7, noisy and local; QX emulator, noiseless and in the cloud). A real device asks for confirmation first.
3. **Setup:** choose the system size, β (0, 0.5, 1, 2, 3, 4 or 5, with the matching temperature shown), the units of the Boltzmann constant, the optimizer and the maximum number of iterations, then pick the qubits on the chip. **Tutorial** walks through every control.
4. **Run:** live cost and fidelity plots, the target and measured density matrices, and the final measurement counts. Runs that measure the fidelity only at the end (every QX run, 3+3 and 4+4) show a Final fidelity card (fidelity, purity and the target's purity) in place of the fidelity plot.

Each run is saved under `~/Documents/QuantumInspireProjects/`. In **Past Experiments**, click a run to view its results again (every plot, the density matrices, counts and outcome, as they were when it finished) or to reuse its settings. **Save results** on the Run screen saves the whole run to a file of your choice, reopened with **Open file…** in Past Experiments, plus every plot (PNG) and its data (CSV) in a folder next to it.

### Backends

| | Starmon-7 | QX emulator | Real hardware (Tuna-9, Tuna-17) |
|---|---|---|---|
| Kind | noisy simulator, on this computer | noiseless simulator, in Quantum Inspire's cloud | real quantum computer |
| Cost evaluations | ideal local simulation | in the cloud, one queued job per iteration | on the device, one queued job per iteration |
| Protocol | chain: β ≈ 0 → 0.5 → 1 → target, each step warm-started | cold start: straight at the target β from small random angles near θ = 0 | cold start |
| Optimizer | COBYLA or SPSA | SPSA (copes with shot noise) | SPSA (copes with gate and shot noise) |
| Final counts | Starmon-7 noise model | simulated, noiseless | measured on the device |

Tuna-5 isn't offered: its star-shaped chip has no 2 × 2 ladder of coupled qubits. The QX emulator allows at most 2048 shots per circuit, so runs there use 2048.

During a cloud run (QX or real hardware) the left sidebar shows the backend's live status (idle, busy, calibrating, benchmarking or offline), and the **Cloud jobs** / **Hardware jobs** tab follows every job from Queued to Results received. **Stop** tries to delete the waiting job from the queue and reports whether Quantum Inspire accepted. A strip along the bottom of the Run screen shows the run's progress: one dot per β stage and the elapsed time.

### System size

| | Qubits | Backends | Fidelity |
|---|---|---|---|
| 2+2 | 4 | all | measured live, every iteration (on the QX emulator: tomography at the end, 9 circuits) |
| 3+3 | 6 | QX emulator, Tuna-17 | tomography at the end (27 circuits) |
| 4+4 | 8 | QX emulator, Tuna-17 | tomography at the end (81 circuits) |

The ansatz needs a 2 × n ladder of coupled qubits: pick the chain A1…An on the chip, and the mirrored chain B is found automatically. On the QX emulator the ladder is fixed to qubits 0…2n−1: QX connects every qubit and reports results by qubit number, so only that layout reads back correctly. For the same reason every size on QX uses circuits that measure each qubit once, at the end.

## Project structure

```
main.py                    app entry point and screen navigation
frontend/                  PySide6 screens and widgets
  home.py                  Home: motivation text and flash cards (flashcards.py, latex.py)
  backend_select.py        backend cards with live availability
  setup.py                 experiment settings and the qubit picker (device_panel.py)
  experiment.py            Run screen and the optimization worker
  hardware_panels.py       device status and Hardware jobs tab
  progress_bar.py          the Run screen's progress strip
  session.py               experiment settings (SessionConfig) and their JSON format
backend/
  backends.py              backend kinds: local simulator, cloud simulator, real hardware
  qi_manager.py            Quantum Inspire login, backends, job submission and cancellation
  starmon7.py              local Starmon-7 emulator
  spsa.py                  SPSA optimizer
  ladder_tfd.py            3+3 / 4+4 experiment on real hardware
  experiment_manager.py    saved experiments (Past Experiments)
gibbs_tfd_reproduction/    2+2 reproduction of the paper: circuit, cost function, tomography
  experiment.py            command-line reproduction run
  experiments/             noisy-optimizer benchmark, reproducibility and sanity-check notebooks
tfd_study/model.py         n+n generalisation (3+3, 4+4): Hamiltonian, ansatz, loss, tomography
Appdata/Img/               chip schematics
```

The command-line scripts in `gibbs_tfd_reproduction/` import their siblings directly, so run them from that folder (for example `cd gibbs_tfd_reproduction && python experiment.py`).

## Code style

Imports are sorted with [isort](https://pycqa.github.io/isort/) using the repository's `.isort.cfg` (standard library, third-party, then this project's modules):

```bash
pip install isort
isort main.py backend frontend tfd_study gibbs_tfd_reproduction
```

## Software stack

- Qiskit and Qiskit Aer
- Quantum Inspire (`qiskit-quantuminspire`, `quantuminspire`, `qi-utilities`)
- NumPy, SciPy and Matplotlib
- PySide6 (Qt for Python)
