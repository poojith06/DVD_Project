#!/bin/bash
# Part B SPICE reference: runs the 18 netlists in netlists/ (one per gate and input; sampling is inside each .cir).
# CSVs go to data/ (one per gate and input, plus one merged file per gate: data/spice_<GATE>.csv, rows sorted by corner, A, B).
# ngspice console output goes to logs/.      Usage (from WSL or any shell with ngspice):  bash run_spice_wsl.sh
cd "$(dirname "$0")/netlists"
mkdir -p ../data ../logs
for f in *.cir; do ngspice -b "$f" > "../logs/${f%.cir}.log" 2>&1; echo "ran $f"; done
for g in INV NAND2 NOR2 AND2 OR2; do
  out=../data/spice_$g.csv
  head -n 1 "$(ls ../data/spice_${g}_*.csv | head -n 1)" > "$out"
  for f in ../data/spice_${g}_*.csv; do tail -n +2 "$f"; done | sort -t, -k1,1n -k2,2n -k3,3n >> "$out"
  echo "$g: $(($(wc -l < "$out") - 1)) rows in data/spice_$g.csv"
done
