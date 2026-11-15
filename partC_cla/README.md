# Part C – 3-bit Carry-Lookahead Adder

SPICE reference for the Part C test circuit (3-bit carry-lookahead adder, 64 input vectors at 200 PVT corners). The gate list that the hierarchical ML prediction needs is `../results/partC/cla3_gates.csv`.
The CLA uses only the Part B gates. Every instance is an INV, NAND2, AND2 or OR2 called as a subcircuit (`X…` lines, as in tut-1 and `Reference/CLA.sp`), and no new gates are designed, so the ML leakage can be built from the Part B gate models.

## Setup

| Item | Value |
|---|---|
| Model file | `22nm_HP.pm` (PTM 22 nm HP, same file as Part A, copy in `netlists/`) |
| Nominal VDD | 1 V |
| Channel length L | 22 nm; widths fixed per gate, all within 1–8 × Lmin |
| Corners | 200 random PVT corners drawn inside ngspice (`set rndseed=16`) |
| Input vectors | A = a2a1a0, B = b2b1b0, all 64 combinations, inputs at exactly 0 or VDD, no carry-in |
| Simulator | ngspice-44.2 (batch, `.op`) |
| Netlist options | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

## How the corners are drawn

As in Part A (`../partA_transistor`), all random values are drawn inside ngspice with `sunif(0)` (uniform in −1 to 1) and applied with `alter` (VDD and input sources), `altermod` (model parameters) and `option temp`. Each corner is one draw of VDD, temperature and process parameters; all NMOS in the adder share one process draw and all PMOS share another (Section 5 of the project spec). VDD is drawn first because the input voltages depend on it. Every corner is then simulated for all 64 input vectors.

| Parameter | Rule | Range |
|---|---|---|
| VDD | nominal ± 10 % | 0.9 – 1.1 V |
| Temperature | uniform | −40 – 125 °C |
| vth0, toxe, ndep, u0, xj | nominal from `22nm_HP.pm` ± 10 %, separate for NMOS and PMOS | same ranges as the Part A README |

Draw order per corner: `vdd, temp, n_vth0, n_toxe, n_ndep, n_u0, n_xj, p_vth0, p_toxe, p_ndep, p_u0, p_xj`. The fixed seed makes the corners reproducible. The drawn corners are also written to `data/corners.csv`. The Part B gate netlists (`../partB_gates`) use the identical corner block, draw order and `rndseed=16`, so Part B and Part C are simulated at **the same 200 corners** (checked against `data/corners.csv`).

**Note:** each process parameter is applied with its own `altermod` line. ngspice applies only the first parameter on a multi-parameter `altermod` line, so writing them all on one line would vary vth0 only (same fix as in the Part A netlists).

## Folder layout

```
partC_cla/
├── netlists/
│   ├── gates.sub               INV, NAND2, NOR2, AND2, OR2 as static CMOS .subckt
│   ├── cla3.sub                .subckt CLA3: gate calls only (includes gates.sub)
│   └── cla3_ngspice.cir        testbench: draws 200 corners x 64 vectors -> ../data/cla3.csv, ../data/corners.csv
│                               (model card: ../../model/22nm_HP.pm)
├── data/
│   ├── cla3.csv                SPICE reference (12,800 rows)
│   └── corners.csv             the 200 drawn corners (one row each)
└── scripts/
    └── regen_cla3.py           re-runs the netlist with the recorded corners into a NEW file and compares with data/cla3.csv
```

The gate-level tables (which gate sees which logic values, and how many gates of each of the 18 cases each vector contains) are not stored as extra data files. The 33 gate instances are listed in `../results/partC/cla3_gates.csv`; `../ml/evaluate_cla.py` computes the logic at every gate input from the A, B vector and sums the Part B gate-model leakages.

## Circuit

The subcircuit port order follows tut-1: `.subckt <GATE> out <inputs> vdd gnd`.

| Signal | Logic | Gates |
|---|---|---|
| Gi, Gi_bar | Gi_bar = NAND2(ai, bi), Gi = INV(Gi_bar) | 3 NAND2 + 3 INV |
| Pi | ai ⊕ bi as four NAND2, with P0 used directly as S0 | 12 NAND2 |
| c1 | G0 | – |
| c2 | G1 + P1·G0 = NAND2(G1_bar, NAND2(P1, G0)) | 2 NAND2 |
| Cout = c3 | (G2 + P2·G1) + (P2·P1)·G0, using only G and P (no ripple through c2) | 2 NAND2 + 2 AND2 + 1 OR2 |
| S1, S2 | P1 ⊕ G0, P2 ⊕ c2, each as four NAND2 | 8 NAND2 |

Each XOR is written out as four NAND2 calls (n1 = NAND(a,b), n2 = NAND(a,n1), n3 = NAND(b,n1), out = NAND(n2,n3)); there is no XOR subcircuit. In total the adder has **27 NAND2, 3 INV, 2 AND2 and 1 OR2 (33 gates, 138 transistors)**.
The reference CLA uses NAND3–NAND5 and a transistor-level XOR. Part B has no models for those, so the wide carry terms here are split into 2-input gates.

## Running

The simulation (about 15–20 minutes for all 12,800 runs):

```bash
cd netlists && ngspice_con -b cla3_ngspice.cir       # writes ../data/cla3.csv and ../data/corners.csv
```

On Windows use `ngspice_con.exe`. Close Excel/OneDrive on `data/` while it runs, or rows can be skipped. Running it overwrites the saved data; the random corner stream of `rndseed=16` depends on the ngspice build, so another build draws other corners.

To regenerate the data without touching the saved files, and independent of the ngspice build:

```bash
python scripts/regen_cla3.py --corners 3 --compare     # 3 corners x 64 vectors, writes ../pipeline_out/partC/cla3_regen.csv
python scripts/regen_cla3.py --compare                 # all 200 corners
```

The script uses the unchanged netlist with the 200 recorded corners from `data/corners.csv` in place of the random draws, and prints the largest relative difference to `data/cla3.csv` (0 for the rows produced in our check).

## Output columns

**`data/cla3.csv`**: `corner, A, B, A2, A1, A0, B2, B1, B0, vdd, temp, n_vth0 … n_xj, p_vth0 … p_xj, leak`, then the voltages of every inter-gate net (`v_g0_bar … v_t4`) and of the outputs (`v_s0, v_s1, v_s2, v_cout`).
`leak` is the total current drawn from the VDD supply (A). The voltages let you check how far each gate input is from an ideal 0 or VDD.

**`data/corners.csv`**: `corner, vdd, temp, n_vth0 … n_xj, p_vth0 … p_xj` for the 200 corners.

The ML prediction for a vector at a corner is the sum over the 33 gates of the Part B gate-model leakage at that corner and at the gate's own input values (`../ml/evaluate_cla.py`):

```
leak_ML(A, B, corner) = Σ_gate  leak_gate_model(case of the gate, corner)
```

## Results (200 corners × 64 vectors = 12,800 simulations)

| Check | Result |
|---|---|
| Rows | 12,800 in `cla3.csv`, 200 in `corners.csv`, no missing values |
| Corner ranges | VDD 0.900 – 1.098 V, T −35.9 – 124.2 °C, all process parameters inside nominal ± 10 % |
| Adder function (S + 8·Cout = A + B, read from SPICE voltages) | correct in all 12,800 rows |
| Total leakage | 0.163 – 24.2 µA, median 2.99 µA |
| Largest distance of any inter-gate net from 0 or VDD | 10.8 mV |
| Sum of Part B SPICE gate leakages vs full CLA SPICE | median error 0.66 %, 95th percentile 2.11 %, max 3.83 %, mean accuracy 99.16 % |

## Check of the gate-sum method

`../ml/evaluate_cla.py` (column `leak_ceiling`) replaces the ML gate models with the **Part B SPICE** gate leakages (`../partB_gates/data/*.csv`, same corners) and compares their sum with the full CLA simulation.
This shows how much error the method itself causes (loading between gates, and gate inputs not sitting exactly at 0 or VDD), separate from any ML error. It is the upper limit for the hierarchical ML prediction.
