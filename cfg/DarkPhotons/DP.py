import awkward as ak
import numpy as np

from cfg.functions.matching import elliptic_match, delta_phi, delta_r
from cfg.functions.utils import set_name
from python.hist_struct import Hist

def np_and(*args):
    out = args[0]
    for arg in args[1:]:
        out = np.bitwise_and(out, arg)
    return out

def define(events, sample_name):

    #isPrompt & isHardProcess & isLastCopy & |pdgId|==11 & final state
    events["GenEle"]=events.GenPart[
                            np_and(
                                (events.GenPart.statusFlags & 0b010000100000001)==8449,
                                np.abs(events.GenPart.pdgId)==11,
                                events.GenPart.status==1)
                            ]
    events["GenEle","mass"]=0.000511
    sortmask=ak.argsort(events.GenEle.pt, axis=1, ascending=False)
    events["GenEle"]=events.GenEle[sortmask]
    set_name(events.GenEle, "GenEle")

    events["Zd"]=events.GenEle[:,0]+events.GenEle[:,1]
    events["Zd","dphi"]=np.abs(delta_phi(events.GenEle[:,0].phi, events.GenEle[:,1].phi))
    events["Zd","deta"]=np.abs(events.GenEle[:,0].eta-events.GenEle[:,1].eta)
    events["Zd","dr"]=delta_r(events.GenEle[:,0].eta, events.GenEle[:,1].eta, events.GenEle[:,0].phi, events.GenEle[:,1].phi)
    events["Zd","dangle"]=events.GenEle[:,0].deltaangle(events.GenEle[:,1])

    events["Lep"]=events.Lep[np.abs(events.Lep.pdgId)==11]
    events["LepGenMatch"] = elliptic_match(
        events.GenEle, events.Lep, ellipse=0.3
    )
    ptmask=events.LepGenMatch.dPt/events.LepGenMatch.Lep.pt<0.5
    events["LepGenMatch"]=events.LepGenMatch[ptmask]

    mindpt_mask=ak.argmin(np.abs(events.LepGenMatch.dPt), axis=2, keepdims=True)
    events["LepGenMatch"]=events.LepGenMatch[mindpt_mask]


    return events


pt_bins = np.linspace(0, 100, 201)
absphi_bins = np.linspace(0, np.pi, 100)
eta_bins = np.linspace(-3, 3, 100)
abseta_bins = np.linspace(0, 3, 100)
r_bins = np.linspace(0, 5, 100)

def get_hists(sample_name):

    return [
        Hist("LepGenMatch/GenEle~pt", bins=pt_bins),
        Hist("GenEle~pt", bins=pt_bins),
        Hist("LepGenMatch/GenEle~eta", bins=eta_bins),
        Hist("GenEle~eta", bins=eta_bins),
        Hist("GenEle~pt[0]", bins=pt_bins),
        Hist("GenEle~pt[1]", bins=pt_bins),
        Hist("Zd~pt", bins=pt_bins),
        Hist("Zd~mass", bins=pt_bins),
        Hist("Zd~dphi", bins=absphi_bins),
        Hist("Zd~deta", bins=abseta_bins),
        Hist("Zd~dr", bins=r_bins),
        Hist("Zd~dangle", bins=absphi_bins),
    ]
