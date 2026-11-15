"""EXP-10: error of the final OFF models at the gate operating points, measured against single-transistor SPICE.

An inverter puts its OFF transistor at a fixed point: INV input 0 -> NMOS OFF with vd = v_out (about VDD), vs = vg = vb = 0;
INV input 1 -> PMOS OFF with vd = v_out (about 0), vs = vg = vb = VDD. For each of the 200 corners the transistor is
simulated alone (every terminal its own source, corner values with one altermod line per parameter, PMOS OFF ig/ib read
from BSIM4's internal currents like the dataset) and compared with the final model of that case. The corresponding test-set
errors from the Part A split are listed next to it. Evaluation only; results/analysis/report_numbers/gate_point_replay.csv
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ml"))
from simlib import OPTIONS, PARAMS, run_folder
from pipeline import TransistorSurrogate

F = REPO / "results"
SIM = HERE / "sim" / "gate_point"
SIM.mkdir(parents=True, exist_ok=True)
for f in SIM.glob("*"):
    f.unlink()
(SIM / "22nm_HP.pm").write_text((REPO / "model" / "22nm_HP.pm").read_text(), newline="\n")
inv = pd.read_csv(REPO / "partB_gates" / "data" / "inv.csv")
BSIM = ("let cig = -(@m1[igs] + @m1[igd] + @m1[igb] + @m1[igcs] + @m1[igcd])\n"
        "let cib = -(@m1[ibd] + @m1[ibs]) + @m1[igb] + @m1[isub] + @m1[igidl] + @m1[igisl]\n")
cases = {"nmos_off": (0, 44e-9, "nmos"), "pmos_off": (1, 88e-9, "pmos")}
bias = {}
for case, (a, w, pol) in cases.items():
    d = inv[inv.A == a].sort_values("corner").reset_index(drop=True)
    vg = np.zeros(len(d)) if pol == "nmos" else d.vdd.to_numpy()
    vs = np.zeros(len(d)) if pol == "nmos" else d.vdd.to_numpy()
    bias[case] = pd.DataFrame({"vdd": d.vdd, "temp": d.temp, "w": w, "vd": d.v_out, "vs": vs, "vg": vg, "vb": vg,
                               **{q: d[f"{pol[0]}_{q}"] for q in PARAMS}, "gate_leak": d.leak})
    t = f"* {case} at the INV operating point\n{OPTIONS}\n.include 22nm_HP.pm\nVd d 0 dc 0\nVg g 0 dc 0\nVs s 0 dc 0\nVb b 0 dc 0\n" \
        f"M1 d g s b {pol} L=22n W={w!r}\n.control\n"
    for i, r in bias[case].iterrows():
        t += f"option temp = {r.temp!r}\n" + "".join(f"altermod {pol} {q} = {r[q]!r}\n" for q in PARAMS)
        t += f"alter vd dc = {r.vd!r}\nalter vg dc = {r.vg!r}\nalter vs dc = {r.vs!r}\nalter vb dc = {r.vb!r}\nop\n{BSIM}"
        t += "print i(vd) i(vs) cig cib\n" if pol == "pmos" else "print i(vd) i(vs) i(vg) i(vb)\n"
    (SIM / f"{case}.cir").write_text(t + ".endc\n.end\n", newline="\n")
print("simulating 400 operating points ...", flush=True)
outs = run_folder(SIM)
rows = []
summary = []
test = pd.read_csv(F / "partA" / "chosen_models_test.csv").set_index("case")
for case in cases:
    vals = [float(m.group(2)) for m in re.finditer(r"(?m)^\s*(i\(vd\)|i\(vs\)|i\(vg\)|i\(vb\)|cig|cib)\s*=\s*([-+0-9.eE]+)", outs[case])]
    sp = np.array(vals).reshape(-1, 4)       # per row: i(vd), i(vs), then (i(vg), i(vb)) for NMOS or (cig, cib) for PMOS
    if cases[case][2] == "nmos":              # same readout as the NMOS dataset: source branch currents
        sp = np.c_[-sp[:, 0], -sp[:, 2], -sp[:, 1], -sp[:, 3]]           # id, ig, is, ib into the terminals
    else:                                     # PMOS OFF dataset: ig, ib from BSIM4 internal currents
        sp = np.c_[-sp[:, 0], sp[:, 2], -sp[:, 1], sp[:, 3]]
    b = bias[case]
    ml = TransistorSurrogate.load(F / "models" / f"{case}.joblib").predict(b.drop(columns="gate_leak")).to_numpy()
    err = 100 * np.abs(ml - sp) / np.abs(sp)
    dom = np.abs(sp[:, [0, 2]]).argmax(1)                                # 0: id is the dominant current, 1: is
    dom_err = np.where(dom == 0, err[:, 0], err[:, 2])
    dom_sp = np.where(dom == 0, np.abs(sp[:, 0]), np.abs(sp[:, 2]))
    dom_ml = np.where(dom == 0, np.abs(ml[:, 0]), np.abs(ml[:, 2]))
    chk = np.abs(dom_sp / b.gate_leak.to_numpy() - 1)
    for i in range(len(b)):
        rows.append({"case": case, "corner": i, "dominant_spice_A": dom_sp[i], "dominant_ml_A": dom_ml[i],
                     "dominant_error_pct": dom_err[i], "signed_error_pct": 100 * (dom_ml[i] - dom_sp[i]) / dom_sp[i]})
    summary.append({"case": case, "corners": len(b),
                    "median_|err|_id_%": np.median(err[:, 0]), "median_|err|_is_%": np.median(err[:, 2]),
                    "median_|err|_ig_%": np.median(err[:, 1]), "median_|err|_ib_%": np.median(err[:, 3]),
                    "median_|ig|_A": np.median(np.abs(sp[:, 1])), "median_|ib|_A": np.median(np.abs(sp[:, 3])),
                    "dominant_current_median_A": np.median(dom_sp), "dominant_current_median_|err|_%": np.median(dom_err), "dominant_p95_|err|_%": np.percentile(dom_err, 95),
                    "dominant_bias_%": np.median(100 * (dom_ml - dom_sp) / dom_sp),
                    "single_FET_vs_gate_SPICE_leak_median_rel": np.median(chk), "single_FET_vs_gate_SPICE_leak_max_rel": chk.max(),
                    "PartA_test_median_%_id": test.loc[case, "id_median_error_pct"], "PartA_test_median_%_is": test.loc[case, "is_median_error_pct"]})
pd.DataFrame(rows).to_csv(F / "analysis" / "report_numbers" / "gate_point_replay.csv", index=False)
S = pd.DataFrame(summary)
S.to_csv(F / "analysis" / "report_numbers" / "gate_point_replay_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S.T.to_string(float_format=lambda x: f"{x:.4g}"))
