"""Decide twirl vs native for 4+4 from the beta = 5 twirling gain on 3+3, rewrite the 4+4 plan.

Gain = mean(F_A, F_B) (paper convention, full readout correction), twirl minus native, at beta = 5 on
8-12-15 / 11-14-16 (twirl: first repeat of the final 3+3 run). Below GAIN_MIN, 4+4 runs native with
NATIVE_REPEATS repeats instead of twirl with 1 repeat (about the same device time per repeat budget).

    python -m tfd_study.choose_twirl_4p4 TWIRL_RUN_DIR NATIVE_RUN_DIR
"""
import json
import sys

GAIN_MIN = 0.05
NATIVE_REPEATS = 3
PLAN = "tfd_study/mitigated_plan_4_chosen.json"


def score(run_dir, mode):
    for r in json.load(open(f"{run_dir}/state.json"))["exp"]["results"].values():
        if r["mode"] == mode and float(r["beta"]) == 5.0 and r["label"].endswith(f"/r0/{mode}"):
            f = r["corrections"]["full"]
            return (f["A"]["fid_paper"] + f["B"]["fid_paper"]) / 2, f["A"]["fid_paper"], f["B"]["fid_paper"]
    raise SystemExit(f"no beta = 5 {mode} result in {run_dir}")


def main():
    tw, nat = score(sys.argv[1], "twirl"), score(sys.argv[2], "native")
    gain = tw[0] - nat[0]
    print(f"beta = 5, 3+3: twirl F_A {tw[1]:.3f} F_B {tw[2]:.3f}; native F_A {nat[1]:.3f} F_B {nat[2]:.3f}; "
          f"gain {gain:+.3f}")
    plan = json.load(open(PLAN))
    if gain < GAIN_MIN:
        plan["run"]["stages"][0].update(modes=["native"], repeats=NATIVE_REPEATS)
        plan["note"] += (f" Twirl gain at beta = 5 on 3+3 was {gain:+.3f} < {GAIN_MIN}: switched to native "
                         f"with {NATIVE_REPEATS} repeats.")
        print(f"4+4: native, {NATIVE_REPEATS} repeats")
    else:
        print("4+4: twirl kept")
    json.dump(plan, open(PLAN, "w"), indent=1)


if __name__ == "__main__":
    main()
