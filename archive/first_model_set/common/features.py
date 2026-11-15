"""Turn the 12 raw inputs into the feature matrix the models see."""
import numpy as np
import pandas as pd

from common.config import ACTIVE_COLUMNS

BOLTZMANN_OVER_Q = 8.617333e-5  # V/K
_MIN_RATIO = 1e-12  # keeps the log finite when vd == vs exactly


def build_features(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Return the 10 active raw inputs plus the physics features.

    vg and vb are used here, in the overdrive and oxide field terms, but are not
    passed on as raw columns because each one is either constant or equal to vdd.
    """
    vd, vs, vg, vb = df["vd"], df["vs"], df["vg"], df["vb"]
    kelvin = df["temp"] + 273.15
    thermal_voltage = BOLTZMANN_OVER_Q * kelvin

    vds = vd - vs
    abs_vds = vds.abs()

    # A MOSFET is symmetric: the electrical source is the lower terminal for NMOS
    # and the higher terminal for PMOS, whichever one the netlist calls "source".
    if polarity == "nmos":
        effective_source = np.minimum(vd, vs)
    else:
        effective_source = np.maximum(vd, vs)
    vgs_eff = vg - effective_source

    features = df[ACTIVE_COLUMNS].copy()
    features["vds"] = vds
    features["abs_vds"] = abs_vds
    features["vgs_eff"] = vgs_eff
    features["vbs_eff"] = vb - effective_source
    features["inv_temp"] = 1.0 / kelvin
    features["phi_sub"] = (vgs_eff.abs() - df["vth0"].abs()) / thermal_voltage
    features["eox_d"] = (vg - vd).abs() / df["toxe"]
    features["eox_s"] = (vg - vs).abs() / df["toxe"]
    features["log_w"] = np.log10(df["w"])
    # log10 of the (1 - exp(-|Vds|/VT)) factor in the subthreshold equation. It holds
    # the sharp dip in log|I| at vd == vs, which is hard for smooth models to learn.
    features["log_vds_term"] = np.log10(
        (-np.expm1(-abs_vds / thermal_voltage)).clip(lower=_MIN_RATIO)
    )
    return features
