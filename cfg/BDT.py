import awkward as ak
import numpy as np
import xgboost as xgb

from cfg.functions.matching import (
    match_obj_to_couple,
    match_obj_to_obj,
    match_to_gen,
)
from cfg.functions.utils import set_name
from python.hist_struct import Hist
from python.inference import xgb_wrapper

BarrelEta = 1.479
model=xgb.Booster()
model.load_model("models/xgboost/BDT_131Xv3.json")

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
features=["CryCluGenMatchAll/"+feat if feat.startswith("CryClu/") else feat for feat in features_minbias]

def define(events, sample_name):
    events["CryClu","showerShape"] = events.CryClu.e2x5/events.CryClu.e5x5
    if sample_name == "MinBias":
        #events = events[:40000]
        events["Tk", "isReal"] = 2
        events = events[ak.num(events.GenEle) == 0]
        events["TkCryCluGenMatchAll"] = match_obj_to_obj(
            events.Tk, events.CryClu, etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
        )
        events["TkCryCluGenMatchAll", "dEtaCryClu"] = events["TkCryCluGenMatchAll", "dEta"]
        events["TkCryCluGenMatchAll", "dPhiCryClu"] = events["TkCryCluGenMatchAll", "dPhi"]
        events["TkCryCluGenMatchAll", "dPtCryClu"] = events["TkCryCluGenMatchAll", "dPt"]
        events["TkCryCluGenMatchAll","BDTscore"]=xgb_wrapper(model, events["TkCryCluGenMatchAll"],features_minbias)
    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatchAll"] = match_to_gen(
            events.CryClu, events.GenEle, etaphi_vars=(("eta", "phi"), ("caloeta", "calophi"))
        )

        set_name(events.CryCluGenMatchAll, "CryCluGenMatchAll")


        #!-------------------Tk-CryClu-Gen Matching-------------------!#

        events["TkCryCluGenMatchAll"] = match_obj_to_couple(
            events.Tk, events.CryCluGenMatchAll, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
        )

        events["TkCryCluGenMatchAll","BDTscore"]=xgb_wrapper(model, events["TkCryCluGenMatchAll"],features)

        events["TkCryCluGenMatchAllReal"]=events["TkCryCluGenMatchAll"][events.TkCryCluGenMatchAll.Tk.isReal==1]
        events["TkCryCluGenMatchAllFake"]=events["TkCryCluGenMatchAll"][events.TkCryCluGenMatchAll.Tk.isReal!=1]


    return events



hists = [Hist("")]
