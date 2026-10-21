"""Variational TFD study for n+n qubits on Tuna-17 (see tfd_study/notebooks)."""

import os

# One thread for numpy's BLAS and OpenMP. By default each process reserves
# per-thread buffers for every core (about 4 GB of committed memory per
# Python process on a 22-core machine), and the studies run many processes
# side by side: the machine runs out of commit space and workers die without
# a Python error. Set before numpy is imported; child processes inherit it.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
