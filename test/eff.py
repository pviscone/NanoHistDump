# %%
import importlib
import sys

sys.path.append("..")

import xgboost as xgb

import python.sample
from cfg.BDT import define

BarrelEta = 1.479
model=xgb.Booster()
model.load_model("/data/pviscone/PhD-notes/submodules/NanoHistDump/models/xgboost/BDT_131Xv3.json")

importlib.reload(python.sample)
Sample = python.sample.Sample

signal_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/DoubleElectron"
pu_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/MinBias"


scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

cryclu_path="CryClu/"
tk_path="Tk/"
couple_path=""
features_minbias=[
    cryclu_path+"standaloneWP",
    cryclu_path+"showerShape",
    cryclu_path+"isolation",
    tk_path+"hitPattern",
    tk_path+"nStubs",
    tk_path+"chi2Bend",
    tk_path+"chi2RPhi",
    tk_path+"chi2RZ",
    couple_path+"dEtaCryClu",
    couple_path+"dPhiCryClu",
    couple_path+"dPtCryClu",
]
features=["CryCluGenMatch/"+feat if feat.startswith("CryClu/") else feat for feat in features_minbias]


signal=Sample("signal", path=signal_path, tree_name="Events", scheme_dict=scheme).events
pu=Sample("pu", path=pu_path, tree_name="Events", scheme_dict=scheme,nevents=40000).events
#%%
signal=define(signal,"DoubleElectrons")
pu=define(pu,"MinBias")

# %%
