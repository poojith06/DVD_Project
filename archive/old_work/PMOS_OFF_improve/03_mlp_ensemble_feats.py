"""ext_features() used by 03_mlp_ensemble.py, importable by the other scripts."""
import numpy as np
from common_setup import *
from common.features import build_features, BOLTZMANN_OVER_Q

def ext_features(d):
    """repo features + per-terminal biases: gate-drain and gate-source voltage, body-drain/body-source voltage, each also in units of Vt"""
    f = build_features(d, "pmos").copy(); vt = BOLTZMANN_OVER_Q * (d["temp"] + 273.15)
    f["vgd"] = d.vg - d.vd; f["vgs"] = d.vg - d.vs; f["vgd_vt"] = f.vgd / vt; f["vgs_vt"] = f.vgs / vt
    f["vmin"] = np.minimum(d.vd, d.vs); f["vds_vt"] = (d.vd - d.vs) / vt; f["sign_vds"] = np.sign(d.vd - d.vs)
    return f

