# Part B – Gate-Level SPICE Reference

SPICE reference for the gate-level validation (step 3): INV, NAND2, NOR2, AND2 and OR2, every input combination, at 200 random PVT corners.
These results are for evaluation only and are not used for training.

## Setup

| Item | Value |
|---|---|
| Model file | `22nm_HP.pm` (PTM 22 nm HP, same file as Part A) |
| Nominal VDD | 1 V |
| Channel length L | 22 nm; widths fixed per gate, all within 1–8 × Lmin |
| Corners | 200 random PVT corners drawn inside ngspice (`set rndseed=16`) |
| Inputs | exactly 0 or VDD (ideal sources) |
| Simulator | ngspice-44.2 (batch, `.op`) |
| Netlist options | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |
| Cases | 18 (INV: 2, each 2-input gate: 4) × 200 corners = 3,600 simulations |

## Folder layout

```
Gates_Part-B/
├── netlists/
│   ├── 22nm_HP.pm          model file
│   ├── inv_ngspice.cir
│   ├── nand2_ngspice.cir
│   ├── nor2_ngspice.cir
│   ├── and2_ngspice.cir
│   └── or2_ngspice.cir
├── data/
│   ├── inv.csv             400 rows (2 inputs x 200 corners)
│   ├── nand2.csv           800 rows (4 inputs x 200 corners)
│   ├── nor2.csv            800 rows
│   ├── and2.csv            800 rows
│   └── or2.csv             800 rows
└── logs/                   ngspice output of each run
```

## How the corners are drawn

As in Part A (`../Datasets`), all random values are drawn inside ngspice with `sunif(0)` (uniform in −1 to 1) and applied with `alter` (VDD and input sources), `altermod` (model parameters) and `option temp`.
A corner is one random draw of VDD, temperature and process parameters. All NMOS in a gate share one draw and all PMOS share another. VDD is drawn first because the input voltages depend on it. Each corner is then simulated for every input combination.

| Parameter | Rule | Range |
|---|---|---|
| VDD | nominal ± 10 % | 0.9 – 1.1 V |
| Temperature | uniform | −40 – 125 °C |
| vth0, toxe, ndep, u0, xj | nominal from `22nm_HP.pm` ± 10 %, separate for NMOS and PMOS | same ranges as the Part A README |

Draw order per corner: `vdd, temp, n_vth0, n_toxe, n_ndep, n_u0, n_xj, p_vth0, p_toxe, p_ndep, p_u0, p_xj`, with `rndseed=16`.
Every gate netlist and the Part C CLA testbench (`../PartC_3bit CLA adder/netlists/cla3_ngspice.cir`) use this identical block. Corner *k* is therefore the same in every Part B file and in Part C, as required by the project spec. The corners are listed in `../PartC_3bit CLA adder/data/corners.csv`.

**Note:** each process parameter is applied with its own `altermod` line. ngspice applies only the first parameter on a multi-parameter `altermod` line, so writing them all on one line would vary vth0 only (same fix as in the Part A netlists).

## Gates

Standard static CMOS. AND2 and OR2 are a NAND2 / NOR2 followed by an INV and are simulated as complete gates.

| Gate | Transistors (W) | Internal nodes recorded |
|---|---|---|
| INV | PMOS 88n, NMOS 44n | – |
| NAND2 | 2 × PMOS 88n (parallel), 2 × NMOS 88n (series) | `v_int` = x (between series NMOS) |
| NOR2 | 2 × PMOS 176n (series), 2 × NMOS 44n (parallel) | `v_int` = y (between series PMOS) |
| AND2 | NAND2 as above + INV (PMOS 88n, NMOS 44n) | `v_int` = x, `v_n` = NAND2 output |
| OR2 | NOR2 as above + INV (PMOS 88n, NMOS 44n) | `v_int` = y, `v_n` = NOR2 output |

## CSV format

Columns: `corner, A, [B], vdd, temp, n_vth0, n_toxe, n_ndep, n_u0, n_xj, p_vth0, p_toxe, p_ndep, p_u0, p_xj, leak, [v_int], [v_n], v_out`

- `leak` is the current drawn from the VDD supply (A).
- `v_int`, `v_n` and `v_out` are the SPICE node voltages (V). Step 4 uses them to set the terminal voltages of each transistor for the ML prediction.

## How to run

ngspice writes the CSVs to `../data/`, so run the netlists from inside `netlists/`.

**Windows (PowerShell):**
```powershell
cd Gates_Part-B\netlists
$ng = "C:\Program Files\Spice64\bin\ngspice_con.exe"
foreach ($g in "inv","nand2","nor2","and2","or2") { & $ng -b "${g}_ngspice.cir" > "..\logs\$g.log" }
```

**Linux / macOS / Git Bash:**
```bash
cd Gates_Part-B/netlists && mkdir -p ../logs
for g in inv nand2 nor2 and2 or2; do ngspice -b ${g}_ngspice.cir > ../logs/$g.log 2>&1; done
```

All five gates together take about a minute.

## Checks

| Check | Result |
|---|---|
| Rows | 400 / 800 / 800 / 800 / 800, no missing values |
| Corners identical to Part C (`corners.csv`) | yes, all files |
| Output logic (v_out vs VDD/2) | correct for every row |
| Independent re-simulation of sample rows from their recorded corner values | identical |
| Sum of these gate leakages vs full Part C CLA SPICE | mean accuracy 99.16 % (see the Part C README) |
