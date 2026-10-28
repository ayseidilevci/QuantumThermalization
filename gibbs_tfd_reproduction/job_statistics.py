"""Quantum Inspire resource use of the 2+2 Tuna-17 experiments: jobs, circuits, shots and time.

Every submission a run collected is in its raw_jobs.jsonl (task kind, circuits returned). Timing per
batch job comes from the Quantum Inspire API: queue wait = queued -> reserved (the backend picks the
job up), on-device = reserved -> finished (execution plus the backend's own overhead). Submissions
that failed and were resubmitted are not in raw_jobs.jsonl, so they are not counted.

Purposes:
  readout calibration   full 16-state calibration (16 circuits = 4 jobs), refreshed every 30 min;
                        4+4: per-qubit calibration (all-0 and all-1, 2 circuits = 1 job)
  SPSA start (kicks)     4+4 only: the best of 8 small random starting points around angles 0
  SPSA gain calibration  3 pairs of points at the start of every beta (12 circuits = 3 jobs)
  SPSA steps             one +/- pair per step (4 circuits = 1 job)
  tomography             36 settings of the final angles (36 circuits = 8 jobs)
  reference tomography   36 settings of the noiseless-optimal angles, at the start and end of a run

    python gibbs_tfd_reproduction/job_statistics.py
Prints one table per experiment and a summary table, and writes experiments/job_statistics/job_statistics.json.
Timings are cached in experiments/job_statistics/.job_timing_cache.json (one API call per job, only once).
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments" / "job_statistics"
CACHE = R / ".job_timing_cache.json"
EXP = ROOT / "experiments"


def run_dir(name):
    """A run directory, under experiments/<experiment>/."""
    return next(EXP.glob(f"*/{name}"))
EXPERIMENTS = {
    "Reproduction chain 0 → 0.5 → 1 → 5": ["parallel_20261019_175824"],
    "Intermediate chain 1 → 2 → 3 → 4 → 5": ["parallel_20261022_140225"],
    "Cold start at β = 5": ["parallel_20261022_215422"],
    "4+4 chain 0.5 → 1 → 2 → 5": ["tfd_n4p1_20261022_053620"],
    "Cold starts at β = 10, 20, 50, 100": ["parallel_20261023_181044", "parallel_20261023_181058",
                                           "parallel_20261023_181113", "parallel_20261023_181128"],
}
PURPOSES = ["readout calibration", "SPSA start (kicks)", "SPSA gain calibration", "SPSA steps", "tomography",
            "reference tomography"]


def purpose(task):
    if task["kind"] == "cal":
        return "readout calibration"
    if task["kind"] == "kick":
        return "SPSA start (kicks)"
    if task["kind"] == "cost":
        return "SPSA steps" if len(task["points"]) == 2 else "SPSA gain calibration"
    return "reference tomography" if task["label"].startswith("reference") else "tomography"


def fetch_timings(ids):
    from compute_api_client import ApiClient, BatchJob, BatchJobsApi, PageBatchJob
    from qi2_shared.client import config
    from qi2_shared.pagination import PageReader

    async def main():
        sem = asyncio.Semaphore(8)
        async with ApiClient(config()) as c:
            async def one(bid):
                async with sem:
                    b = await PageReader[PageBatchJob, BatchJob]().get_single(
                        BatchJobsApi(c).read_batch_jobs_batch_jobs_get, id=int(bid))
                    return bid, [b.queued_at.isoformat(), b.reserved_at.isoformat(), b.finished_at.isoformat()]
            return dict(await asyncio.gather(*(one(i) for i in ids)))
    return asyncio.run(main())


def timings(all_ids):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [i for i in all_ids if i not in cache]
    if missing:
        print(f"fetching {len(missing)} job timings from Quantum Inspire ...")
        cache.update(fetch_timings(missing))
        CACHE.write_text(json.dumps(cache))
    return {i: [datetime.fromisoformat(t) for t in cache[i]] for i in all_ids}


def load(run):
    st = json.loads((run_dir(run) / "state.json").read_text())
    rows = [json.loads(line) for line in (run_dir(run) / "raw_jobs.jsonl").read_text().splitlines() if line.strip()]
    return st["cfg"], rows


def main():
    runs = {run: load(run) for runs in EXPERIMENTS.values() for run in runs}
    tim = timings(sorted({r["job_id"] for _, rows in runs.values() for r in rows}))
    out = {}
    for name, run_list in EXPERIMENTS.items():
        agg = defaultdict(lambda: {"jobs": 0, "circuits": 0, "shots": 0, "queue_s": [], "device_s": []})
        starts, ends = [], []
        for run in run_list:
            cfg, rows = runs[run]
            for r in rows:
                p = purpose(r["task"])
                n = len(r["counts"])
                q, s, f = tim[r["job_id"]]
                a = agg[p]
                a["jobs"] += 1
                a["circuits"] += n
                a["shots"] += n * (cfg["shots_tomo"] if r["task"]["kind"] == "tomo" else cfg["shots_cost"])
                a["queue_s"].append((s - q).total_seconds())
                a["device_s"].append((f - s).total_seconds())
                starts.append(q)
                ends.append(f)
        res = {}
        for p in PURPOSES + ["total"]:
            if p == "total":
                parts = [agg[k] for k in PURPOSES if k in agg]
                a = {"jobs": sum(x["jobs"] for x in parts), "circuits": sum(x["circuits"] for x in parts),
                     "shots": sum(x["shots"] for x in parts), "queue_s": sum((x["queue_s"] for x in parts), []),
                     "device_s": sum((x["device_s"] for x in parts), [])}
            elif p in agg:
                a = agg[p]
            else:
                continue
            res[p] = {"jobs": a["jobs"], "circuits": a["circuits"], "shots": a["shots"],
                      "device_h": sum(a["device_s"]) / 3600, "device_median_s": float(np.median(a["device_s"])),
                      "queue_median_min": float(np.median(a["queue_s"])) / 60}
        res["wall_clock_h"] = (max(ends) - min(starts)).total_seconds() / 3600
        out[name] = res

        print(f"\n== {name}  ({', '.join(run_list)}); wall clock {res['wall_clock_h']:.1f} h")
        print(f"{'purpose':24s} {'jobs':>5s} {'circuits':>9s} {'shots':>11s} {'on-device':>10s} "
              f"{'per job':>8s} {'queue/job':>10s}")
        for p in PURPOSES + ["total"]:
            if p in res:
                x = res[p]
                print(f"{p:24s} {x['jobs']:>5d} {x['circuits']:>9d} {x['shots']:>11,d} {x['device_h']:>8.2f} h "
                      f"{x['device_median_s']:>6.0f} s {x['queue_median_min']:>7.1f} min")
    (R / "job_statistics.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", R / "job_statistics.json")


if __name__ == "__main__":
    main()
