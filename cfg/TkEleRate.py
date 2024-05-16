import awkward as ak
import numpy as np

from python.hist_struct import Hist

BarrelEta=1.479

def define(events, sample_name):

    if sample_name == "MinBias":
        #events = events[:40000]

        #!-------------------TkEle -------------------!#
        events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<BarrelEta]
        events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
        mask_tight_ele = 0b0010
        events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
        events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]


    return events


pt_bins = np.linspace(0,120,121)
eta_bins = np.linspace(-2,2,50)
bdt_bins = np.linspace(0,1.01,102)

hists = [
        #TkEle
        Hist("TkEle","pt",bins=pt_bins,fill_mode="rate_vs_ptcut"),



    ]
