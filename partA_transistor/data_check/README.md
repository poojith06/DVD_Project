# Deliverable 3: data check of the four Part A datasets

Produced by `validation_checks/data_check.py` (log: `validation_checks/data_check.txt`) from
`partA_transistor/data/{nmos_on,nmos_off,pmos_on,pmos_off}.csv` (10,000 rows each).

| File | Content |
|---|---|
| `kcl_check.csv` | Kirchhoff check per case |
| `trend_off_vs_temperature.png` | OFF current vs temperature (NMOS OFF, PMOS OFF) |
| `trend_off_vs_vth0.png` | OFF current vs vth0 (NMOS OFF) and \|vth0\| (PMOS OFF) |
| `trend_ig_vs_toxe.png` | gate current \|ig\| vs toxe (all four cases) |
| `trend_numbers.csv` | numbers behind the trend statements below |

**Definition used in the plots:** OFF current = max(\|id\|, \|is\|). The drain and source currents swap roles when
vs > vd, so the larger of the two is the channel (leakage) current in either orientation. Plots: faint scatter of all
10,000 rows, log y-axis, red line = median in 20 equal-width bins.

## 1. Kirchhoff check, id + ig + is + ib (currents are into the terminals, so the sum must be 0)

| Case | median \|sum\| (A) | max \|sum\| (A) | median rel. | p95 rel. | max rel. | rows with rel. > 1e-3 |
|---|---|---|---|---|---|---|
| NMOS ON | 8.18e-12 | 9.87e-10 | 5.13e-07 | 4.22e-06 | 2.84e-02 | 15 |
| NMOS OFF | 2.59e-17 | 8.29e-13 | 8.50e-07 | 3.87e-06 | 8.96e-06 | 0 |
| PMOS ON | 4.20e-14 | 1.00e-10 | 4.85e-09 | 1.57e-06 | 1.36e-02 | 18 |
| PMOS OFF | 1.00e-19 | 9.16e-13 | 1.14e-06 | 5.52e-06 | 9.91e-06 | 0 |

rel. = \|sum\| / max terminal \|I\| of the row. Median residuals are about 1e-6 of the largest current (the 6-digit CSV
precision and ngspice's reltol = 1e-6); the 95th percentile is below 6e-6 in every case.

* **PMOS OFF has no larger residual.** Because `ig` and `ib` come from BSIM4's own computed currents (not from the
  source branch currents, which are rounded to steps of about 2.2e-16 A), the residual stays at the 1e-6 level even
  for rows whose largest current is 1e-15 to 1e-14 A (median \|sum\| 3.8e-21 A there). Largest relative residual 9.91e-06.
* **The only larger residuals are in the ON cases:** 15 NMOS ON and 18 PMOS ON rows have a relative residual above
  1e-3 (maximum 2.84e-02 and 1.36e-02). In all of them the largest terminal current is tiny (at most 1.42e-13 A for
  NMOS ON and 6.31e-14 A for PMOS ON, medians 6.3e-14 A and 1.8e-14 A): the "ON" device has its source close to VDD
  (NMOS) or to 0 V (PMOS) and is barely on. Their absolute residuals are at most 7.7e-16 A (NMOS ON) and 1.4e-16 A
  (PMOS ON), i.e. up to about 3 times the ~2.2e-16 A resolution of the branch-current readout that `ig`/`ib` use in
  the ON datasets; the relative residual is large only because the currents themselves are that small.

## 2. Trends

Spearman = rank correlation of log10(I) with the parameter over all 10,000 rows (every other parameter is random too,
so the correlation is diluted). The controlled effect is the change of log10(I) in decades for the stated step, from a
linear regression of log10(I) on all inputs (so the other parameters are held fixed).

| Plot | Case | Spearman | Binned median first to last bin | Controlled effect |
|---|---|---|---|---|
| OFF current vs temperature | NMOS OFF | +0.172 | 1.67e-11 to 3.70e-11 A (x2.2) | +0.386 decades per 100 C |
| OFF current vs temperature | PMOS OFF | +0.397 | 1.32e-14 to 3.08e-12 A (x233) | +1.25 decades per 100 C |
| OFF current vs \|vth0\| | NMOS OFF | -0.082 | 3.07e-11 to 1.72e-11 A (x0.56) | -0.138 decades per 50 mV |
| OFF current vs \|vth0\| | PMOS OFF | -0.091 | 2.65e-13 to 3.44e-14 A (x0.13) | -0.304 decades per 50 mV |
| \|ig\| vs toxe | NMOS ON | -0.375 | 6.53e-11 to 3.24e-12 A (x0.050) | -0.705 decades per 0.1 nm |
| \|ig\| vs toxe | NMOS OFF | -0.602 | 8.06e-11 to 2.88e-12 A (x0.036) | -0.748 decades per 0.1 nm |
| \|ig\| vs toxe | PMOS ON | -0.192 | 3.94e-12 to 1.97e-13 A (x0.050) | -0.488 decades per 0.1 nm |
| \|ig\| vs toxe | PMOS OFF | -0.699 | 8.88e-15 to 5.24e-17 A (x0.0059) | -1.08 decades per 0.1 nm |

All three trends have the physically expected sign in every case: the OFF current rises with temperature and falls
with the threshold voltage, and the gate (tunnelling) current falls steeply with oxide thickness (a factor 20 to 170
across the 0.945 to 1.155 nm range).

**Why the vth0 and temperature trends look weak over all rows (NMOS OFF in particular).** The OFF current
max(\|id\|, \|is\|) contains parts that do not depend on vth0 or depend little on temperature: the drain and source
junction leakage and the gate-induced drain leakage (GIDL). In the scatter plot of PMOS OFF this is visible as a floor
at about 1e-14 A. To check that the subthreshold channel current behaves as expected, the same regression was repeated
on the rows with the source within 50 mV of its rail and the other terminal more than 0.5 V away (the situation of a
leaking transistor in a gate, rows where the channel current dominates):

| Subset | Rows | Spearman with T | Effect of +100 C | Effect of +50 mV vth0 |
|---|---|---|---|---|
| NMOS OFF, source < 50 mV from 0, other > 0.5 V | 498 | +0.562 | +0.899 decades | -0.551 decades |
| PMOS OFF, source < 50 mV from VDD, other > 0.5 V | 514 | +0.456 | +0.81 decades | -0.593 decades |

In this regime the OFF current rises by about 0.8 to 0.9 decades per 100 C and falls by 0.55 to 0.59 decades per
50 mV of vth0 (a slope of about 85 to 90 mV per decade, as expected for subthreshold conduction with an ideality factor
of about 1.4). The data therefore show the right physics; the dilution over all rows comes from the junction and GIDL
parts of the leakage.
