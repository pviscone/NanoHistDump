# %%
import importlib
import sys

sys.path.append("..")

import awkward as ak
import numpy as np

import python.sample
from cfg.functions.matching import match_to_gen, select_match

importlib.reload(python.sample)
Sample = python.sample.Sample

fname="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU0"

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

BarrelEta = 1.479
events = Sample("", path=fname, tree_name="Events", scheme_dict=scheme).events


events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]

#%%
events["CCGenAll"]=match_to_gen(events.CryClu,events.GenEle,etaphi_vars=(("eta","phi"),("caloeta","calophi")),dr_cut=0.4)

#%%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

from matplotlib.colors import LogNorm


bins=[np.linspace(-0.3,0.3,100),np.linspace(-0.1,0.1,100)]
eta=np.array(ak.flatten(events.CCGenAll.dEta))
phi=np.array(ak.flatten(events.CCGenAll.dPhi))

plt.hist2d(phi,eta,bins=bins,norm=LogNorm(),density=True)
plt.xlabel(r"$\Delta\phi$")
plt.ylabel(r"$\Delta\eta$")
plt.colorbar()
plt.title("Crystal Cluster - Gen Electron")
plt.grid()