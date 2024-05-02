import awkward as ak
import numpy as np
import xgboost as xgb

from cfg.functions.matching import match_obj_to_couple, match_obj_to_obj, match_to_gen
from cfg.functions.utils import set_name
from python.hist_struct import Hist
from python.inference import xgb_wrapper

BarrelEta = 1.479
model=xgb.Booster()
model.load_model("/data/pviscone/PhD-notes/submodules/NanoHistDump/models/xgboost/BDT_131Xv3.json")

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

#score_cuts=np.concatenate([np.linspace(0.5,0.8,10),np.linspace(0.8,0.9,15)])
def define(events, sample_name):
    #add_collection(events, "triggers")
    events["CryClu","showerShape"] = events.CryClu.e2x5/events.CryClu.e5x5
    if sample_name == "MinBias":
        events = events[:40000]

        events["Tk", "isReal"] = 2
        events = events[ak.num(events.GenEle) == 0]

        events["TkCryCluGenMatch"] = match_obj_to_obj(
            events.CryClu, events.Tk, etaphi_vars=(("eta", "phi"), ("caloEta", "caloPhi")),nested=True
        )
        events["TkCryCluGenMatch", "dEtaCryClu"] = events["TkCryCluGenMatch", "dEta"]
        events["TkCryCluGenMatch", "dPhiCryClu"] = events["TkCryCluGenMatch", "dPhi"]
        events["TkCryCluGenMatch", "dPtCryClu"] = events["TkCryCluGenMatch", "dPt"]
        events["TkCryCluGenMatch","BDTscore"]=xgb_wrapper(model, events["TkCryCluGenMatch"],features_minbias,nested=True)

        maxbdt_mask=ak.argmax(events["TkCryCluGenMatch"].BDTscore,axis=2,keepdims=True)
        events["TkCryCluGenMatch"]=ak.flatten(events["TkCryCluGenMatch"][maxbdt_mask],axis=2)
    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatch"] = match_to_gen(
            events.GenEle, events.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")),nested=True)

        mindpt_mask=ak.argmin(np.abs(events["CryCluGenMatch"].dPt),axis=2,keepdims=True)

        events["CryCluGenMatch"]=ak.flatten(events["CryCluGenMatch"][mindpt_mask],axis=2)

        set_name(events.CryCluGenMatch, "CryCluGenMatch")


        #!-------------------Tk-CryClu-Gen Matching-------------------!#

        events["TkCryCluGenMatch"] = match_obj_to_couple(
            events.Tk, events.CryCluGenMatch, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi")),nested=True
        )

        events["TkCryCluGenMatch","BDTscore"]=xgb_wrapper(model, events["TkCryCluGenMatch"],features,nested=True)


        #!-------------------BDT selection-------------------!#
        maxbdt_mask=ak.argmax(events["TkCryCluGenMatch"].BDTscore,axis=2,keepdims=True)
        events["TkCryCluGenMatch"]=ak.flatten(events["TkCryCluGenMatch"][maxbdt_mask],axis=2)


    #!-------------------TkEle-Gen Matching-------------------!#
    events["TkEleGenMatch"] = match_to_gen(
        events.GenEle, events.Tk, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")),nested=True
    )
    mindpt_mask=ak.argmin(np.abs(events["TkEleGenMatch"].dPt),axis=2,keepdims=True)
    events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask],axis=2)

    """
    #! You should move the cuts to the hist struct, not creating collections in loops
    for cut in score_cuts:
        events["TkCryCluGenMatchAll",f"isBDT{cut*1000:.0f}"]=events["TkCryCluGenMatchAll","BDTscore"]>=cut

        #For rate vs threshold cut
        trigger_cut=ak.max(events["TkCryCluGenMatchAll",f"isBDT{cut*1000:.0f}"],axis=1)

        events["triggers",f"isBDT{cut*1000:.0f}"]=trigger_cut
        events["triggers",f"isElectron{cut*1000:.0f}"]=ak.any(ak.mask(events,trigger_cut==1).TkCryCluGenMatchAll.Tk.isReal==1,axis=1)

        #max because they are all equal and avoid me to deal with missing values
        if sample_name != "MinBias":
            events["triggers",f"genPt{cut*1000:.0f}"]=ak.max(ak.mask(events,trigger_cut==1).TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.pt,axis=1)
            events["triggers",f"genEta{cut*1000:.0f}"]=ak.max(ak.mask(events,trigger_cut==1).TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.eta,axis=1)

    events["triggers","maxBDT"]=ak.max(events["TkCryCluGenMatchAll","BDTscore"],axis=1)
    """

    if sample_name != "MinBias":
        events["TkCryCluGenMatchReal"]=events["TkCryCluGenMatch"][events.TkCryCluGenMatch.Tk.isReal==1]
        events["TkCryCluGenMatchFake"]=events["TkCryCluGenMatch"][events.TkCryCluGenMatch.Tk.isReal!=1]
    return events

pt_bins = np.linspace(1,101,50)
eta_bins = np.linspace(-2,2,50)
bdt_bins = np.array([0,0.4,0.6,0.8,1.01])

hists = [#signal
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryCluGenMatch/GenEle","pt",bins=[bdt_bins,pt_bins]),
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryCluGenMatch/GenEle","eta",bins=[bdt_bins,eta_bins]),

        Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","eta",bins=eta_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu","pt",bins=pt_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu","eta",bins=eta_bins),
        Hist("GenEle","pt",bins=pt_bins),
        Hist("GenEle","eta",bins=eta_bins),
        Hist("TkEleGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkEleGenMatch/GenEle","eta",bins=eta_bins),
        #bkg
        #Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryClu","pt",bins=[bdt_bins,pt_bins]),
        #Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryClu","eta",bins=[bdt_bins,eta_bins]),
        #Hist("TkCryCluGenMatch/CryClu","pt",bins=pt_bins),
        #Hist("TkCryCluGenMatch/CryClu","eta",bins=eta_bins),

        #all
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/Tk","isReal",bins=[bdt_bins,np.array([0,1,2,3])]),
        Hist("TkCryCluGenMatch","BDTscore",bins=bdt_bins),
        Hist("TkCryCluGenMatch/Tk","isReal",bins=np.array([0,1,2,3])),
    ]
""" Hist("GenEle","eta",**eta_bins),
Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","pt",**pt_bins),
Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","eta",**eta_bins),
Hist("TkCryCluGenMatch/Tk","isReal",hist_range=[0,3],bins=3),
Hist("TkCryCluGenMatch","BDTscore",**bdt_bins), """
