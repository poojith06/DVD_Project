# Part A – Single-Transistor Datasets

SPICE datasets used to train the transistor-level leakage models (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF).
Every sample is a single-FET DC operating-point simulation in ngspice with randomly drawn PVT and bias values.

## Setup

| Item | Value |
|---|---|
| Model file | `22nm_HP.pm` (PTM 22 nm High Performance, BSIM4 level 54) |
| Nominal VDD | 1 V |
| Channel length L | 22 nm (fixed) |
| Simulator | ngspice-44.2 (batch mode, `.op` analysis) |
| Samples | 10,000 per case |
| Netlist options | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` |

## Folder layout

```
Datasets/
├── netlists/
│   ├── 22nm_HP.pm               model file
│   ├── nmos_on_ngspice.cir
│   ├── nmos_off_ngspice.cir
│   ├── pmos_on_ngspice.cir
│   └── pmos_off_ngspice.cir
└── data/
    ├── nmos_on.csv
    ├── nmos_off.csv
    ├── pmos_on.csv
    └── pmos_off.csv
```

## How the parameters are varied

All parameters are drawn **independently and uniformly** for every sample, with VDD drawn first because the Vd and Vs ranges depend on it. The random draws are done inside ngspice with `sunif(0)` (uniform in −1 to 1) and applied with `alter` (sources, width), `altermod` (model parameters) and `option temp` (temperature).

| Parameter | Rule | Range |
|---|---|---|
| VDD | nominal ± 10 % | 0.9 – 1.1 V |
| Vd, Vs | uniform 0 to VDD | 0 – VDD of that sample |
| W | 1 × Lmin to 8 × Lmin | 22 – 176 nm |
| Temperature | uniform | −40 – 125 °C |

Process parameters (nominal from `22nm_HP.pm`, ± 10 %, separate for NMOS and PMOS):

| Parameter | NMOS nominal | NMOS range | PMOS nominal | PMOS range |
|---|---|---|---|---|
| vth0 (V) | 0.50308 | 0.4528 – 0.5534 | −0.4606 | −0.5067 – −0.4145 |
| toxe (m) | 1.05e-9 | 0.945e-9 – 1.155e-9 | 1.1e-9 | 0.99e-9 – 1.21e-9 |
| ndep (cm⁻³) | 5.5e18 | 4.95e18 – 6.05e18 | 4.4e18 | 3.96e18 – 4.84e18 |
| u0 (m²/Vs) | 0.04 | 0.036 – 0.044 | 0.0095 | 0.00855 – 0.01045 |
| xj (m) | 7.2e-9 | 6.48e-9 – 7.92e-9 | 7.2e-9 | 6.48e-9 – 7.92e-9 |

Gate and body voltages are fixed by the case:

| Case | Vg | Vb | Random seed |
|---|---|---|---|
| NMOS ON | VDD | 0 | 13 |
| NMOS OFF | 0 | 0 | 12 |
| PMOS ON | 0 | VDD | 14 |
| PMOS OFF | VDD | VDD | 15 |

Each case uses its own fixed seed (`set rndseed=` in the netlist), so the datasets are reproducible and independent of each other.

## CSV format

Columns: `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib`

- **Inputs (12):** vdd … xj. Units: V, °C, m, V, V, V, V, V, m, cm⁻³, m²/Vs, m.
- **Outputs (4):** id, ig, is, ib in A.
- **Sign convention:** each current is the current flowing **into** that transistor terminal, so `id + ig + is + ib ≈ 0` (Kirchhoff check).
- Values are written with 6 significant digits.

## How to run

ngspice writes the CSV into the folder it is run from, so run the netlists from inside `Datasets/netlists` and move the CSVs to `data/` afterwards.

**Windows (PowerShell):**
```powershell
cd Datasets\netlists
$ng = "C:\path\to\Spice64\bin\ngspice_con.exe"
& $ng -b nmos_on_ngspice.cir  | Out-Null
& $ng -b nmos_off_ngspice.cir | Out-Null
& $ng -b pmos_on_ngspice.cir  | Out-Null
& $ng -b pmos_off_ngspice.cir | Out-Null
Move-Item *.csv ..\data\ -Force
```

**Linux / macOS:**
```bash
cd Datasets/netlists
for c in nmos_on nmos_off pmos_on pmos_off; do ngspice -b ${c}_ngspice.cir > /dev/null; done
mv *.csv ../data/
```

Each case takes from a few seconds to a couple of minutes. Each CSV should have 10,001 lines (header + 10,000 rows).

## Notes

- Use `ngspice_con.exe` on Windows; `ngspice.exe` opens a GUI window.
- The repeated warning *"toxe, toxp and dtox all given … dtox ignored"* is expected (toxe is being varied) and harmless.
- If ngspice prints `Permission denied`, another program (Excel, OneDrive sync, antivirus) locked the CSV and rows were skipped. Close Excel or pause OneDrive and re-run; the fixed seed reproduces the same data.
- PMOS OFF currents go down to ~1e-15–1e-16 A, close to the simulator's numerical floor, so its Kirchhoff residual is larger than for the other cases.
