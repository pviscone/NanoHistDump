# %%
import sys

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.interpolate import PchipInterpolator
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append("../../..")

from cfg.functions.matching import match_to_gen, select_match
from python.sample import Sample

hep.style.use("CMS")


save = False
tag = "131Xv3"

signal_path = "/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/DoubleElectron"
pu_path = "/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/MinBias"

schema = {
    "CaloEGammaCrystalClustersGCT": "CryClu",
    "DecTkBarrel": "Tk",
    "GenEl": "GenEle",
}

pu = Sample("", path=pu_path, tree_name="Events", scheme_dict=schema).events
signal = Sample("", path=signal_path, tree_name="Events", scheme_dict=schema).events

BarrelEta = 1.479

signal["GenEle"] = signal.GenEle[np.abs(signal.GenEle.eta) < BarrelEta]
signal = signal[ak.num(signal.GenEle) > 0]


cc_gen = match_to_gen(signal.CryClu, signal.GenEle, etaphi_vars=(("eta", "phi"), ("caloeta", "calophi")))
cc_gen = select_match(cc_gen, cc_gen.GenEle.idx)


#%%
from hepstats.modeling import bayesian_blocks

fig, (ax1, ax2) = plt.subplots(2, 1)
ptPU_array = ak.flatten(ak.drop_none(pu.CryClu.pt))
bins=bayesian_blocks(ptPU_array)
centers = (bins[1:] + bins[:-1]) / 2
ptPU = ax1.hist(ptPU_array, bins=bins, density=True, histtype="step", label="PU", linewidth=2)[0]

ptSignal_array = ak.flatten(ak.drop_none(cc_gen.CryClu.pt))
ptSignal = ax1.hist(ptSignal_array, bins=bins, density=True, histtype="step", label="signal", linewidth=2)[0]



ax1.legend()
ax1.grid()
ax1.set_yscale("log")
ax1.set_ylabel("Density")

ratio = ptSignal/ptPU
smoothed = np.exp(lowess(exog=centers, endog=np.log(1e-18 + ratio), frac=0.175, it=50)[:, 1])

interp_func = PchipInterpolator(centers, np.log(smoothed))


def interp(func, x):
    res = np.zeros_like(x)
    res[x > 100] = func(100)
    res[x < 100] = func(x[x < 100])
    return res

ax1.set_xlim(-10,130)
ax2.set_xlim(-10,130)
x = np.linspace(0, 120, 500)
plt.stairs(ratio,edges=bins, color="black", label="Signal/PU", linewidth=2)


ax2.plot(centers, smoothed, ".", color="red", label="smoothed", markersize=10)
ax2.plot(x, np.exp(interp(interp_func, x)), color="dodgerblue", label="interpolated", linewidth=2)
ax2.set_yscale("log")
ax2.grid()
ax2.legend()

ax2.set_xlabel("CryClu Pt [GeV]")
ax2.set_ylabel("SF")

hep.cms.text("Simulation Phase-2", ax=ax1, loc=0)
fig.savefig("fig/ptSF.pdf")
# %%

sf = np.concatenate([x[:, None], np.exp(interp(interp_func, x))[:, None]], axis=1)

if save:
    np.save(f"ptSF_{tag}.npy", sf)

# %%
