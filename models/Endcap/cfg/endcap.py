import pathlib

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

classes=2

path=pathlib.Path(__file__).parent.parent.resolve().joinpath(f"endcap_131Xv3.json")
model.load_model(path)

def np_and(*args):
    out = args[0]
    for arg in args[1:]:
        out = np.bitwise_and(out, arg)
    return out

features=[
    "HGCalClu_coreshowerlength",
    "HGCalClu_meanz",
    "HGCalClu_spptot",
    "HGCalClu_seetot",
    "HGCalClu_szz",
    "HGCalClu_multiClassPionIdScore",
    "HGCalClu_multiClassEmIdScore",
    "Tk_PtFrac",
    "PtRatio",
    "abs_dEta",
    "abs_dPhi",
]

features_signal=["HGCalCluGenMatch_"+feat if feat.startswith("HGCalClu") else feat for feat in features]


def define(events, sample_name):
    hgcal_mask=np_and(
        np.abs(events["HGCalClu","eta"])<2.4,
        events["HGCalClu","multiClassPuIdScore"]<=0.4878136,
        events["HGCalClu","multiClassEmIdScore"]>0.115991354
    )
    events["HGCalClu"]=events["HGCalClu"][hgcal_mask]

    #!-------------------TkEle -------------------!#
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)>BarrelEta]
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<=2.4]
    events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]


    if "MinBias" in sample_name:

        events = events[ak.num(events.GenEle) == 0]

        events["TkHGCalCluMatch"] = elliptic_match(
            events.HGCalClu, events.Tk, etaphi_vars=[["eta", "phi"], ["caloEta", "caloPhi"]], ellipse=ellipse
        )
        events["TkHGCalCluMatch","Tk","PtFrac"] = events.TkHGCalCluMatch.Tk.pt/ak.sum(events.TkHGCalCluMatch.Tk.pt,axis=2)

        events["TkHGCalCluMatch","BDTscore"]=xgb_wrapper(model, events["TkHGCalCluMatch"],features,nested=True,layout_template=events.TkHGCalCluMatch.PtRatio.layout)

        maxbdt_mask=ak.argmax(events["TkHGCalCluMatch"].BDTscore,axis=2,keepdims=True)
        events["TkHGCalCluMatch"]=ak.flatten(events["TkHGCalCluMatch"][maxbdt_mask],axis=2)


    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) >= BarrelEta]
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < 2.4]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------HGCalClu-Gen Matching-------------------!#
        events["HGCalCluGenMatch"] = match_to_gen(
            events.GenEle, events.HGCalClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")), nested=True
        )

        mindpt_mask=ak.argmin(np.abs(events["HGCalCluGenMatch"].dPt),axis=2,keepdims=True)

        events["HGCalCluGenMatch"]=ak.flatten(events["HGCalCluGenMatch"][mindpt_mask],axis=2)

        set_name(events.HGCalCluGenMatch, "HGCalCluGenMatch")


        #!-------------------Tk-HGCalClu-Gen Matching-------------------!#

        events["TkHGCalCluGenMatch"] = elliptic_match(
            events.HGCalCluGenMatch,
            events.Tk,
            etaphi_vars=[["HGCalClu/eta", "HGCalClu/phi"], ["caloEta", "caloPhi"]],
            ellipse=ellipse,
        )

        events["TkHGCalCluGenMatch","Tk","PtFrac"] = events.TkHGCalCluGenMatch.Tk.pt/ak.sum(events.TkHGCalCluGenMatch.Tk.pt,axis=2)

        events["TkHGCalCluGenMatch","BDTscore"]=xgb_wrapper(model, events["TkHGCalCluGenMatch"],features_signal,nested=True,layout_template=events.TkHGCalCluGenMatch.PtRatio.layout)


        #!-------------------BDT selection-------------------!#
        maxbdt_mask=ak.argmax(events["TkHGCalCluGenMatch"].BDTscore,axis=2,keepdims=True)
        events["TkHGCalCluGenMatch"]=ak.flatten(events["TkHGCalCluGenMatch"][maxbdt_mask],axis=2)


        #!-------------------TkEle-Gen Matching-------------------!#
        events["TkEleGenMatch"] = match_to_gen(
            events.GenEle, events.TkEle, etaphi_vars=(("caloeta", "calophi"), ("caloEta", "caloPhi")),nested=True
        )
        mindpt_mask=ak.argmin(np.abs(events["TkEleGenMatch"].dPt),axis=2,keepdims=True)
        events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask],axis=2)

        #events["TkHGCalCluGenMatchReal"]=events["TkHGCalCluGenMatch"][events.TkHGCalCluGenMatch.Tk.isReal==1]
        #events["TkHGCalCluGenMatchFake"]=events["TkHGCalCluGenMatch"][events.TkHGCalCluGenMatch.Tk.isReal!=1]

    return events

pt_bins = np.linspace(0,120,121)
eta_bins = np.linspace(-2,2,50)
bdt_bins = np.linspace(0,1.01,102)

hists = [#signal
        Hist("TkHGCalCluGenMatch","BDTscore","TkHGCalCluGenMatch/HGCalCluGenMatch/GenEle","pt",bins=[bdt_bins,pt_bins,pt_bins],additional_axes=[["TkHGCalCluGenMatch/HGCalCluGenMatch/HGCalClu","pt"]]),
        Hist("TkHGCalCluGenMatch","BDTscore","TkHGCalCluGenMatch/HGCalCluGenMatch/GenEle","eta",bins=[bdt_bins,eta_bins, pt_bins],additional_axes=[["TkHGCalCluGenMatch/HGCalCluGenMatch/HGCalClu","pt"]]),
        Hist("HGCalCluGenMatch/GenEle","pt",bins=pt_bins),

        Hist("TkHGCalCluGenMatch/HGCalCluGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkHGCalCluGenMatch/HGCalCluGenMatch/GenEle","eta",bins=eta_bins),
        Hist("TkHGCalCluGenMatch/HGCalCluGenMatch/HGCalClu","pt",bins=pt_bins),
        Hist("TkHGCalCluGenMatch/HGCalCluGenMatch/HGCalClu","eta",bins=eta_bins),
        Hist("GenEle","pt",bins=pt_bins),
        Hist("GenEle","eta",bins=eta_bins),
        Hist("TkEleGenMatch/GenEle","pt",bins=pt_bins),
        Hist("TkEleGenMatch/GenEle","eta",bins=eta_bins),
        Hist("TkHGCalCluGenMatch","BDTscore","TkHGCalCluGenMatch/Tk","isReal",bins=[bdt_bins,np.array([0,1,2,3])]),
        Hist("TkHGCalCluGenMatch","BDTscore",bins=bdt_bins),
        Hist("TkHGCalCluGenMatch/Tk","isReal",bins=np.array([0,1,2,3])),

        #bkg
        Hist("TkHGCalCluMatch/HGCalClu","pt","TkHGCalCluMatch","BDTscore",bins=[pt_bins,bdt_bins],fill_mode="rate_pt_vs_score"),
        Hist("TkHGCalCluMatch/HGCalClu","pt",bins=pt_bins),
        Hist("TkHGCalCluMatch","BDTscore",bins=bdt_bins),

        #TkEle
        Hist("TkEle","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),

        #HGCalClu
        Hist("HGCalClu","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),


    ]
