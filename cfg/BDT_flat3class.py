import awkward as ak
import numpy as np
import xgboost as xgb

from cfg.functions.matching import elliptic_match, match_to_gen
from cfg.functions.utils import set_name
from python.hist_struct import Hist
from python.inference import xgb_wrapper

ellipse = [[0.03, 0.3]]
BarrelEta = 1.479
model=xgb.Booster()
model.load_model("/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/flat3class_131Xv3.json")


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
    "dEta",
    "dPhi",
    "PtRatio",
    "nMatch"
]

features_signal=["CryCluGenMatch_"+feat if feat.startswith("CryClu") else feat for feat in features]


def define(events, sample_name):
    #!-------------------TkEle -------------------!#
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<BarrelEta]
    events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]
    events["CryClu","ss"] = events.CryClu.e2x5/events.CryClu.e5x5
    events["CryClu","relIso"] = events.CryClu.isolation/events.CryClu.pt
    if sample_name == "MinBias":

        events = events[ak.num(events.GenEle) == 0]

        events["TkCryCluMatch"] = elliptic_match(
            events.CryClu, events.Tk, etaphi_vars=[["eta", "phi"], ["caloEta", "caloPhi"]], ellipse=ellipse
        )

        events["TkCryCluMatch","nMatch"]=ak.num(events.TkCryCluMatch.Tk.pt,axis=2)

        events["TkCryCluMatch","BDTscore"]=xgb_wrapper(model, events["TkCryCluMatch"],features,nested=True,layout_template=events.TkCryCluMatch.PtRatio.layout)

        maxbdt_mask=ak.argmax(events["TkCryCluMatch"].BDTscore,axis=2,keepdims=True)
        events["TkCryCluMatch"]=ak.flatten(events["TkCryCluMatch"][maxbdt_mask],axis=2)

    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatch"] = match_to_gen(
            events.GenEle, events.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")), nested=True
        )

        mindpt_mask=ak.argmin(np.abs(events["CryCluGenMatch"].dPt),axis=2,keepdims=True)

        events["CryCluGenMatch"]=ak.flatten(events["CryCluGenMatch"][mindpt_mask],axis=2)

        set_name(events.CryCluGenMatch, "CryCluGenMatch")


        #!-------------------Tk-CryClu-Gen Matching-------------------!#

        events["TkCryCluGenMatch"] = elliptic_match(
            events.CryCluGenMatch,
            events.Tk,
            etaphi_vars=[["CryClu/eta", "CryClu/phi"], ["caloEta", "caloPhi"]],
            ellipse=ellipse,
        )

        events["TkCryCluGenMatch","nMatch"]=ak.num(events.TkCryCluGenMatch.Tk.pt,axis=2)

        events["TkCryCluGenMatch","BDTscore"]=xgb_wrapper(model, events["TkCryCluGenMatch"],features_signal,nested=True,layout_template=events.TkCryCluGenMatch.PtRatio.layout)


        #!-------------------BDT selection-------------------!#
        maxbdt_mask=ak.argmax(events["TkCryCluGenMatch"].BDTscore,axis=2,keepdims=True)
        events["TkCryCluGenMatch"]=ak.flatten(events["TkCryCluGenMatch"][maxbdt_mask],axis=2)


        #!-------------------TkEle-Gen Matching-------------------!#
        events["TkEleGenMatch"] = match_to_gen(
            events.GenEle, events.TkEle, etaphi_vars=(("caloeta", "calophi"), ("caloEta", "caloPhi")),nested=True
        )
        mindpt_mask=ak.argmin(np.abs(events["TkEleGenMatch"].dPt),axis=2,keepdims=True)
        events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask],axis=2)

        #events["TkCryCluGenMatchReal"]=events["TkCryCluGenMatch"][events.TkCryCluGenMatch.Tk.isReal==1]
        #events["TkCryCluGenMatchFake"]=events["TkCryCluGenMatch"][events.TkCryCluGenMatch.Tk.isReal!=1]

    return events

pt_bins = np.linspace(0,120,121)
eta_bins = np.linspace(-2,2,50)
bdt_bins = np.linspace(0,1.01,102)

hists = [#signal
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryCluGenMatch/GenEle","pt",bins=[bdt_bins,pt_bins,pt_bins],additional_axes=[["TkCryCluGenMatch/CryCluGenMatch/CryClu","pt"]]),
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/CryCluGenMatch/GenEle","eta",bins=[bdt_bins,eta_bins, pt_bins],additional_axes=[["TkCryCluGenMatch/CryCluGenMatch/CryClu","pt"]]),
        Hist("CryCluGenMatch/GenEle","pt",bins=pt_bins),

        Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle","eta",bins=eta_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu","pt",bins=pt_bins),
        Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu","eta",bins=eta_bins),
        Hist("GenEle","pt",bins=pt_bins),
        Hist("GenEle","eta",bins=eta_bins),
        Hist("TkEleGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkEleGenMatch/GenEle","eta",bins=eta_bins),
        Hist("TkCryCluGenMatch","BDTscore","TkCryCluGenMatch/Tk","isReal",bins=[bdt_bins,np.array([0,1,2,3])]),
        Hist("TkCryCluGenMatch","BDTscore",bins=bdt_bins),
        Hist("TkCryCluGenMatch/Tk","isReal",bins=np.array([0,1,2,3])),

        #bkg
        Hist("TkCryCluMatch/CryClu","pt","TkCryCluMatch","BDTscore",bins=[pt_bins,bdt_bins],fill_mode="rate_pt_vs_score"),
        Hist("TkCryCluMatch/CryClu","pt",bins=pt_bins),
        Hist("TkCryCluMatch","BDTscore",bins=bdt_bins),

        #TkEle
        Hist("TkEle","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),

        #CryClu
        Hist("CryClu","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),


    ]
