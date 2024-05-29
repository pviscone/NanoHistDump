import awkward as ak
import numpy as np

from cfg.functions.matching import match_to_gen
from cfg.functions.utils import set_name
from python.hist_struct import Hist

BarrelEta=1.479

def define(events, sample_name):


    #events = events[:40000]
    #!-------------------TkEle -------------------!#
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<BarrelEta]
    events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]

    if sample_name!="MinBias":
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]
        #!-------------------TkEle-Gen Matching-------------------!#
        events["TkEleGenMatch"] = match_to_gen(
            events.GenEle, events.TkEle, etaphi_vars=(("caloeta", "calophi"), ("caloEta", "caloPhi")),nested=True
        )
        mindpt_mask=ak.argmin(np.abs(events["TkEleGenMatch"].dPt),axis=2,keepdims=True)
        events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask],axis=2)


        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatch"] = match_to_gen(
            events.GenEle, events.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")),nested=True)

        mindpt_mask=ak.argmin(np.abs(events["CryCluGenMatch"].dPt),axis=2,keepdims=True)

        events["CryCluGenMatch"]=ak.flatten(events["CryCluGenMatch"][mindpt_mask],axis=2)

        set_name(events.CryCluGenMatch, "CryCluGenMatch")



    return events


pt_bins = np.linspace(0,120,121)
eta_bins = np.linspace(-2,2,50)
bdt_bins = np.linspace(0,1.01,102)

hists = [
        #TkEle
        Hist("TkEle","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),
        Hist("CryClu","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),
        Hist("TkEleGenMatch/GenEle","pt",bins=pt_bins),
        Hist("CryCluGenMatch/GenEle","pt",bins=pt_bins),
        Hist("GenEle","pt",bins=pt_bins),



    ]
