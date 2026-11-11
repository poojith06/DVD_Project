# Part C – 3-bit Carry-Lookahead Adder

SPICE reference for the Part C test circuit, plus the gate-level input tables that the hierarchical ML prediction needs.
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

As in Part A (`../Datasets`), all random values are drawn inside ngspice with `sunif(0)` (uniform in −1 to 1) and applied with `alter` (VDD and input sources), `altermod` (model parameters) and `option temp`. Each corner is one draw of VDD, temperature and process parameters; all NMOS in the adder share one process draw and all PMOS share another (Section 5 of the project spec). VDD is drawn first because the input voltages depend on it. Every corner is then simulated for all 64 input vectors.

| Parameter | Rule | Range |
|---|---|---|
| VDD | nominal ± 10 % | 0.9 – 1.1 V |
| Temperature | uniform | −40 – 125 °C |
| vth0, toxe, ndep, u0, xj | nominal from `22nm_HP.pm` ± 10 %, separate for NMOS and PMOS | same ranges as the Part A README |

Draw order per corner: `vdd, temp, n_vth0, n_toxe, n_ndep, n_u0, n_xj, p_vth0, p_toxe, p_ndep, p_u0, p_xj`. The fixed seed makes the corners reproducible. The drawn corners are also written to `data/corners.csv`. The Part B gate netlists (`../Gates_Part-B`) use the identical corner block, draw order and `rndseed=16`, so Part B and Part C are simulated at **the same 200 corners** (checked against `data/corners.csv`).

**Note:** each process parameter is applied with its own `altermod` line. ngspice applies only the first parameter on a multi-parameter `altermod` line, so writing them all on one line would vary vth0 only (same fix as in the Part A netlists).

## Folder layout

```
PartC_3bit CLA adder/
├── gen_cla3_gate_inputs.py     logic at every gate input for each vector (step 7) + gate-sum check
├── run_partC.sh                runs everything in order
├── netlists/
│   ├── 22nm_HP.pm              model file (same as Part A)
│   ├── gates.sub               INV, NAND2, NOR2, AND2, OR2 as static CMOS .subckt
│   ├── cla3.sub                .subckt CLA3: gate calls only (includes gates.sub)
│   └── cla3_ngspice.cir        testbench: draws 200 corners x 64 vectors -> ../data/cla3.csv, ../data/corners.csv
├── data/
│   ├── cla3.csv                SPICE reference (12,800 rows)
│   ├── corners.csv             the 200 drawn corners (one row each)
│   ├── cla3_gate_inputs.csv    one row per (vector, gate instance), 64 x 33 rows
│   └── cla3_gate_counts.csv    one row per vector: number of gates in each of the 18 gate cases
└── logs/cla3.log
```

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

```bash
bash run_partC.sh
# or step by step
cd netlists && ngspice_con -b cla3_ngspice.cir > ../logs/cla3.log && cd ..
python gen_cla3_gate_inputs.py            # uses ../Gates_Part-B/data for the gate-sum check (run Part B first)
```

On Windows use `ngspice_con.exe` (e.g. `"C:\Program Files\Spice64in
gspice_con.exe"`). The full run takes about 15–20 minutes. Close Excel/OneDrive on `data/` while it runs, or rows can be skipped.

## Output columns

**`data/cla3.csv`**: `corner, A, B, A2, A1, A0, B2, B1, B0, vdd, temp, n_vth0 … n_xj, p_vth0 … p_xj, leak`, then the voltages of every inter-gate net (`v_g0_bar … v_t4`) and of the outputs (`v_s0, v_s1, v_s2, v_cout`).
`leak` is the total current drawn from the VDD supply (A). The voltages let you check how far each gate input is from an ideal 0 or VDD.

**`data/corners.csv`**: `corner, vdd, temp, n_vth0 … n_xj, p_vth0 … p_xj` for the 200 corners.

**`data/cla3_gate_counts.csv`**: for each vector, the number of gates in each case `INV_0 … OR2_11`. The ML prediction for a vector at a corner is

```
leak_ML(A, B, corner) = Σ_case  count(A, B, case) × leak_gate_model(case, corner)
```

**`data/cla3_gate_inputs.csv`**: the same information per gate instance (instance name, gate type, input and output logic values, and net names). Use it to look at errors gate by gate.

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

`gen_cla3_gate_inputs.py` replaces the ML gate models with the **Part B SPICE** gate leakages (`../Gates_Part-B/data/*.csv`, same corners) and compares their sum with the full CLA simulation.
This shows how much error the method itself causes (loading between gates, and gate inputs not sitting exactly at 0 or VDD), separate from any ML error. It is the upper limit for the hierarchical ML prediction.
