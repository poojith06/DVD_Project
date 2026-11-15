#!/bin/bash
cd "$(dirname "$0")/netlists"
ngspice -b pmos_off_hires_ngspice.cir > ../run.log 2>&1
wc -l ../data/pmos_off_hires_raw.csv
