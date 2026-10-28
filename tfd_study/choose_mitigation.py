"""Pick the circuit mode (and 3+3 placement) from the beta = 1 step-by-step tests, write the final plans.

Score = mean of F_A and F_B (paper convention, full readout correction). Platform and native cost the
same number of jobs; twirl costs ~4x as many (4 variants per setting), so it is chosen only when it beats
the best of the other two by at least TWIRL_MARGIN. The mode is decided on the averaged score of all
placements tested; the 3+3 placement is the one that scores best in that mode. The 4+4 placement is fixed
(4-8-12-15 / 7-11-14-16, the ladder that contains the best 3+3 placement 8-12-15 / 11-14-16).

    python -m tfd_study.choose_mitigation RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np

TWIRL_MARGIN = 0.01
BETAS = [0.5, 1.0, 2.0, 5.0]
LAYOUT_4 = [4, 8, 12, 15, 7, 11, 14, 16]
BASE = {"backend": "Tuna-17", "p": 1, "varsigma": 1.57, "twirls": 4, "shots_tomo": 4096,
        "shots_cal": 8192, "max_inflight": 5}


def main():
    scores = defaultdict(dict)                     # layout -> mode -> score
    for d in sys.argv[1:]:
        for r in json.load(open(f"{d}/state.json"))["exp"]["results"].values():
            f = r["corrections"]["full"]
            scores[tuple(r["layout"])][r["mode"]] = (f["A"]["fid_paper"] + f["B"]["fid_paper"]) / 2
    modes = set.intersection(*(set(m) for m in scores.values()))
    mean = {m: float(np.mean([scores[lay][m] for lay in scores])) for m in modes}
    for lay, s in scores.items():
        print("-".join(map(str, lay)), {m: round(v, 3) for m, v in sorted(s.items())})
    print("mean over placements", {m: round(v, 3) for m, v in sorted(mean.items())})

    cheap = max((m for m in modes if m != "twirl"), key=mean.get)
    mode = "twirl" if "twirl" in modes and mean["twirl"] >= mean[cheap] + TWIRL_MARGIN else cheap
    layout3 = list(max(scores, key=lambda lay: scores[lay][mode]))
    print(f"chosen: mode {mode}, 3+3 placement {layout3}")

    rep4 = 1 if mode == "twirl" else 2
    for n, layout, reps, seed in ((3, layout3, 2, 11), (4, LAYOUT_4, rep4, 12)):
        plan = {"note": f"{n}+{n} final run, mode '{mode}' and placement chosen by choose_mitigation.py from the "
                        f"beta = 1 step-by-step tests ({', '.join(sys.argv[1:])}).",
                "run": dict(BASE, n=n, seed=seed, stages=[
                    {"layouts": [layout], "betas": BETAS, "modes": [mode], "repeats": reps}])}
        path = f"tfd_study/mitigated_plan_{n}_chosen.json"
        json.dump(plan, open(path, "w"), indent=1)
        print(f"wrote {path}: {reps} repeat(s)")


if __name__ == "__main__":
    main()
