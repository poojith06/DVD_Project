"""Re-simulate a few rows of every SPICE netlist (after the folder restructuring) and compare with the saved CSVs.
The netlist is copied into validation_checks/sim/resim/ (same relative layout, so ../../model/22nm_HP.pm and ../data resolve),
and the lines that draw random numbers (sunif) are replaced by the values recorded in the CSV, because the random stream of
rndseed depends on the ngspice build. Everything else (circuit, options, model card, altermod lines, current read-out) is the
unchanged netlist. Part A: first 5 rows of each CSV; Parts B/C: first 3 corners (all input combinations).
Needs ngspice in WSL (distribution 'Ubuntu'). Output: validation_checks/netlist_resim_check.txt"""
import re, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO=Path(__file__).resolve().parents[1]
S=REPO/"validation_checks"/"sim"/"resim"
if S.exists(): shutil.rmtree(S)
S.mkdir(parents=True); (S/"model").mkdir(parents=True); shutil.copy(REPO/"model/22nm_HP.pm",S/"model")
A_MAP={"vdd":"vdd","t":"temp","w":"w","vd":"vd","vs":"vs","vg":"vg","vb":"vb","p1":"vth0","p2":"toxe","p3":"ndep","p4":"u0","p5":"xj"}
C_MAP={"vdd":"vdd","t":"temp","n1":"n_vth0","n2":"n_toxe","n3":"n_ndep","n4":"n_u0","n5":"n_xj","p1":"p_vth0","p2":"p_toxe","p3":"p_ndep","p4":"p_u0","p5":"p_xj"}
def wsl(p): return "/mnt/c/"+str(p).replace("\\","/")[3:]
def prep(src,depth,mp,idx,nvals,table,loopvar,loopcount_re,outname):
    """copy netlist into scratch tree <part>/netlists, substitute sunif draws by vectors"""
    rel=src.relative_to(REPO)
    dst=S/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    (S/rel.parts[0]/"data").mkdir(parents=True,exist_ok=True)
    if rel.parts[0]=="partC_cla" : pass
    if "readout" in str(rel): (S/rel.parts[0]/"pmos_off_readout_check"/"data").mkdir(parents=True,exist_ok=True)
    for f in src.parent.glob("*.sub"): shutil.copy(f,dst.parent)
    t=src.read_text()
    vecs=[]; out=[]
    for ln in t.splitlines():
        m=re.match(r"\s*let (\w+) = .*sunif\(0\)",ln)
        if m and m.group(1) in mp:
            v=m.group(1); xs=[float(x) for x in table[mp[v]][:nvals]]
            neg=all(x<0 for x in xs)
            if (not neg) and any(x<0 for x in xs):
                vecs.append(f"compose vec_{v} values "+" ".join(repr(x+1000.0) for x in xs)); out.append(f"  let {v} = vec_{v}[{loopvar}] - 1000"); continue
            vals=" ".join(repr(abs(x) if neg else x) for x in xs)
            vecs.append(f"compose vec_{v} values {vals}"); out.append(f"  let {v} = {'-' if neg else ''}vec_{v}[{loopvar}]"); continue
            continue
        out.append(ln)
    t="\n".join(out)+"\n"
    t=loopcount_re(t)
    # put the vectors right after '.control'
    t=t.replace(".control\n",".control\n"+"\n".join(vecs)+"\n",1)
    dst.write_text(t,newline="\n")
    return dst
def run(dst):
    r=subprocess.run(["wsl","-d","Ubuntu","--","bash","-c",f"cd '{wsl(dst.parent)}' && OMP_WAIT_POLICY=passive ngspice -b {dst.name} > {dst.stem}.out 2>&1"],capture_output=True,text=True)
    return (dst.parent/f"{dst.stem}.out").read_text(errors="replace")
def cmp(new,old,cols,floor=1e-18):
    res={}
    for c in cols:
        a=new[c].to_numpy(float); b=old[c].to_numpy(float)
        res[c]=float(np.max(np.abs(a-b)/np.maximum(np.abs(b),floor)))
    return res
rows=[]
NR=5
for case in ["nmos_on","nmos_off","pmos_on","pmos_off"]:
    src=REPO/f"partA_transistor/netlists/{case}_ngspice.cir"
    ref=pd.read_csv(REPO/f"partA_transistor/data/{case}.csv")
    dst=prep(src,0,A_MAP,"k",NR,ref,"k",lambda t:t.replace("let n = 10000","let n = %d"%NR),None)
    log=run(dst)
    new=pd.read_csv(dst.parent.parent/"data"/f"{case}.csv")
    r=cmp(new,ref.iloc[:NR],["id","ig","is","ib"]); r2=cmp(new,ref.iloc[:NR],["vdd","temp","w","vd","vs","vg","vb"],1e-30)
    big=[ (np.abs(ref.iloc[:NR][c])>1e-12) for c in ["id","ig","is","ib"]]
    rows.append((case,NR,max(r.values()),max(r2.values()),r))
# readout-check netlist
src=REPO/"partA_transistor/pmos_off_readout_check/netlists/pmos_off_ngspice.cir"
ref=pd.read_csv(REPO/"partA_transistor/data/pmos_off.csv")
dst=prep(src,0,A_MAP,"k",NR,ref,"k",lambda t:t.replace("let n = 10000","let n = %d"%NR),None)
run(dst)
new=pd.read_csv(dst.parent.parent/"data"/"pmos_off.csv")
r=cmp(new,ref.iloc[:NR],["id","ig","is","ib"]); rows.append(("pmos_off(readout_check netlist)",NR,max(r.values()),0,r))
# gates and cla
NC=3
corn=pd.read_csv(REPO/"partC_cla/data/corners.csv")
for g in ["inv","nand2","nor2","and2","or2"]:
    src=REPO/f"partB_gates/netlists/{g}_ngspice.cir"
    ref=pd.read_csv(REPO/f"partB_gates/data/{g}.csv")
    dst=prep(src,0,C_MAP,"c",NC,corn,"c",lambda t:re.sub(r"let ncorner = \d+","let ncorner = %d"%NC,t),None)
    run(dst)
    new=pd.read_csv(dst.parent.parent/"data"/f"{g}.csv")
    k=["corner","A"]+(["B"] if "B" in ref else [])
    n=len(new); old=ref.iloc[:n]
    assert (new[k].to_numpy()==old[k].to_numpy()).all()
    cols=["leak","v_out"]+(["v_int"] if "v_int" in ref else [])
    r=cmp(new,old,cols); rows.append((g,n,max(r.values()),0,r))
src=REPO/"partC_cla/netlists/cla3_ngspice.cir"
ref=pd.read_csv(REPO/"partC_cla/data/cla3.csv")
dst=prep(src,0,C_MAP,"c",NC,corn,"c",lambda t:re.sub(r"let ncorner = \d+","let ncorner = %d"%NC,t),None)
run(dst)
new=pd.read_csv(dst.parent.parent/"data"/"cla3.csv"); n=len(new); old=ref.iloc[:n]
assert (new[["corner","A","B"]].to_numpy()==old[["corner","A","B"]].to_numpy()).all()
cols=["leak","v_cout","v_s0","v_g0","v_t4"]
r=cmp(new,old,cols); rows.append(("cla3",n,max(r.values()),0,r))
lines=[]
for name,n,m,m2,r in rows:
    lines.append(f"{name:32s} rows {n:4d}  max rel diff {m:.2e}  (inputs {m2:.1e})  "+" ".join(f"{k}={v:.1e}" for k,v in r.items()))
    print(lines[-1])
verdict = "RESIM: PASS" if all(x[2] <= 1e-5 for x in rows) else "RESIM: FAIL"
print(verdict)
(REPO / "validation_checks" / "netlist_resim_check.txt").write_text(chr(10).join(lines) + chr(10) + verdict + chr(10))
