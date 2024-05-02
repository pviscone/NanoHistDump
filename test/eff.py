# %%
import importlib
import sys

sys.path.append("..")

import awkward as ak
import numpy as np
import xgboost as xgb

import python.sample
from cfg.functions.matching import (
    match_obj_to_couple,
    match_obj_to_obj,
    match_to_gen,
)
from cfg.functions.utils import set_name
from python.inference import xgb_wrapper

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
signal["CryClu","showerShape"] = signal.CryClu.e2x5/signal.CryClu.e5x5

#!-------------------GEN Selection-------------------!#
signal["GenEle"] = signal.GenEle[np.abs(signal.GenEle.eta) < BarrelEta]
signal = signal[ak.num(signal.GenEle) > 0]

#!-------------------CryClu-Gen Matching-------------------!#
signal["CryCluGenMatch"] = match_to_gen(
    signal.GenEle, signal.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")),nested=True)

mindpt_mask=ak.argmin(np.abs(signal["CryCluGenMatch"].dPt),axis=2,keepdims=True)

signal["CryCluGenMatch"]=ak.flatten(signal["CryCluGenMatch"][mindpt_mask],axis=2)

set_name(signal.CryCluGenMatch, "CryCluGenMatch")


#!-------------------Tk-CryClu-Gen Matching-------------------!#

signal["TkCryCluGenMatch"] = match_obj_to_couple(
    signal.Tk, signal.CryCluGenMatch, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi")),nested=True
)

signal["TkCryCluGenMatch","BDTscore"]=xgb_wrapper(model, signal["TkCryCluGenMatch"],features,nested=True)


#!-------------------BDT selection-------------------!#
maxbdt_mask=ak.argmax(signal["TkCryCluGenMatch"].BDTscore,axis=2,keepdims=True)
signal["TkCryCluGenMatch"]=ak.flatten(signal["TkCryCluGenMatch"][maxbdt_mask],axis=2)
# %%
#!-------------------Min Bias-------------------!#
pu["CryClu","showerShape"] = pu.CryClu.e2x5/pu.CryClu.e5x5

pu["Tk", "isReal"] = 2
pu = pu[ak.num(pu.GenEle) == 0]

pu["TkCryCluGenMatch"] = match_obj_to_obj(
    pu.CryClu, pu.Tk, etaphi_vars=(("eta", "phi"), ("caloEta", "caloPhi")),nested=True
)
pu["TkCryCluGenMatch", "dEtaCryClu"] = pu["TkCryCluGenMatch", "dEta"]
pu["TkCryCluGenMatch", "dPhiCryClu"] = pu["TkCryCluGenMatch", "dPhi"]
pu["TkCryCluGenMatch", "dPtCryClu"] = pu["TkCryCluGenMatch", "dPt"]
pu["TkCryCluGenMatch","BDTscore"]=xgb_wrapper(model, pu["TkCryCluGenMatch"],features_minbias,nested=True)

maxbdt_mask=ak.argmax(pu["TkCryCluGenMatch"].BDTscore,axis=2,keepdims=True)
pu["TkCryCluGenMatch"]=ak.flatten(pu["TkCryCluGenMatch"][maxbdt_mask],axis=2)
#%%

from python.plotters import TEfficiency

pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]")
