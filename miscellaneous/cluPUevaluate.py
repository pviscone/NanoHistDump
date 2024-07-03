# %%
import importlib
import sys

sys.path.append("..")

import awkward as ak
import numpy as np

import python.sample
from cfg.functions.matching import match_to_gen, elliptic_match
from python.inference import xgb_wrapper
importlib.reload(python.sample)
Sample = python.sample.Sample

fname="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU200"

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

BarrelEta = 1.479
events = Sample("", path=fname, tree_name="Events", scheme_dict=scheme).events


events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]

#%%
events["Tk"]=events.Tk[events.Tk.isReal==1]
events["CryCluGenAll"]=match_to_gen(events.CryClu,events.GenEle,etaphi_vars=(("eta","phi"),("caloeta","calophi")),dr_cut=0.4)


#%%
events["CryCluGenAll"]=events["CryCluGenAll"][(events["CryCluGenAll"].dEta/0.025)**2+(events["CryCluGenAll"].dPhi/0.2)**2>1]

events["CryCluTk"]=elliptic_match(events.CryCluGenAll.CryClu,events.Tk,etaphi_vars=(("eta","phi"),("caloEta","caloPhi")),ellipse=[0.03,0.3])


#%%



features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_isIso",
    "CryClu_isSS",
    "CryClu_isLooseTkIso",
    "CryClu_isLooseTkSS",
    "CryClu_brems",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_hitPattern",
    #"Tk_pt",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RZ",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "dEta",
    "dPhi",
    "PtRatio",
    "nMatch"
]

events["CryCluTk","Tk","PtFrac"] = events.CryCluTk.Tk.pt/ak.sum(events.CryCluTk.Tk.pt,axis=2)
events["CryCluTk","CryClu","ss"] = events.CryCluTk.CryClu.e2x5/events.CryCluTk.CryClu.e5x5
events["CryCluTk","CryClu","relIso"] = events.CryCluTk.CryClu.isolation/events.CryCluTk.CryClu.pt
events["CryCluTk","nMatch"]=ak.num(events.CryCluTk.Tk.pt,axis=2)

import xgboost as xgb
model=xgb.Booster()
model.load_model("/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/flat3class_131Xv3.json")

events["CryCluTk","BDTscore"]=xgb_wrapper(model, events["CryCluTk"],features,nested=True,layout_template=events.CryCluTk.PtRatio.layout)


#%%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

bdt=ak.flatten(events["CryCluTk"].BDTscore,axis=2)
bdt=ak.flatten(bdt)
plt.hist(bdt,density=True)
plt.grid()
plt.xlabel("BDT score")