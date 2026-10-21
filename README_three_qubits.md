# Quantum Thermalization

Quick-start guide for running the three-spin thermofield-double (TFD) experiment.

## Requirements

- Python environment with the project dependencies installed.
- A Quantum Inspire account with access to the selected backend.

From the project root, authenticate if needed:

```bash
qi login
```

## Run an experiment

Example: run on **Tuna-17** with readout mitigation and final A-register tomography enabled:

```bash
python gibbs_tfd_reproduction/extension_experiment.py \
    --backend-name "Tuna-17" \
    --betas 1 \
    --varsigma 2.05 \
    --p-layers 1 \
    --optimizer COBYLA \
    --shots 2048 \
    --maxiter 100 \
    --rhobeg 0.8 \
    --tol 0.01 \
    --readout-mitigation \
    --calibration-shots 2048 \
    --calibration-interval 10 \
    --tomography \
    --tomography-shots 2048 \
    --outdir outputs/three_tuna_17_mitigated
```

To use the emulator, replace `Tuna-17` with `QX emulator`.

### Main options

| Option | Description |
|---|---|
| `--backend-name` | Backend to run on, e.g. `Tuna-17` or `QX emulator` |
| `--betas` | Inverse temperature value(s) |
| `--varsigma` | TFD protocol parameter |
| `--p-layers` | Number of variational layers |
| `--optimizer` | Classical optimizer, e.g. `COBYLA` |
| `--shots` | Shots per objective measurement circuit |
| `--maxiter` | Maximum optimizer iterations |
| `--readout-mitigation` | Enable readout-error mitigation |
| `--calibration-shots` | Shots used for readout calibration |
| `--calibration-interval` | Calibration interval in objective evaluations |
| `--tomography` | Run final A-register tomography |
| `--tomography-shots` | Shots per tomography setting |
| `--outdir` | Directory for experiment outputs |

Hardware jobs may take time: each objective evaluation submits two circuits, and tomography uses 27 measurement settings per beta.

## Generate plots

After the experiment finishes, generate plots from the saved results:

```bash
python gibbs_tfd_reproduction/plot_3spin.py \
    --outdir outputs/three_tuna_17_mitigated
```

If needed, check the plotting script's supported options:

```bash
python gibbs_tfd_reproduction/plot_3spin.py --help
```

## Results and output directories

- Raw experiment data and results are saved in the directory specified by `--outdir`.
- Plot generation is handled separately by `plot_3spin.py`.
- Use a different output directory for each run to keep experiments separate.
- The experiment script may clear an existing output directory unless `--keep-output` is specified. Use `--keep-output` when you want to preserve existing results.

For all supported experiment options:

```bash
python gibbs_tfd_reproduction/extension_experiment.py --help
```
