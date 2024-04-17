import awkward as ak
import numpy as np

from cfg.functions.matching import (
    count_idx_dpt,
    count_idx_pt,
    count_idx_pt_isReal,
    match_obj_to_couple,
    match_to_gen,
    select_match,
)
from cfg.functions.utils import add_collection, set_name
from python.hist_struct import Hist

BarrelEta = 1.479


def define(events):
    add_collection(events, "n")
    #!-------------------GEN Selection-------------------!#
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
    # events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
    events = events[ak.num(events.GenEle) > 0]
    print(f"Number of events after selection: {len(events)}")
    print(f"Number of GenEle {ak.sum(ak.num(events.GenEle,axis=1))}")

    #!-------------------CryClu-Gen Matching-------------------!#
    events["CryCluGenMatchAll"] = match_to_gen(events.CryClu, events.GenEle, etaphi_vars=(("eta", "phi"), ("caloeta", "calophi")))
    set_name(events.CryCluGenMatchAll, "CryCluGenMatchAll")

    events["CryCluGenMatch"] = select_match(events.CryCluGenMatchAll, events.CryCluGenMatchAll.GenEle.idx)

    events["n", "CryCluGenMatchAll"], events["n", "CryCluGenMatchAllPt"] = count_idx_pt(
        events.CryCluGenMatchAll.GenEle.idx,
        events.GenEle.pt,
    )

    _, events["n", "CryCluGenMatchAllMinAbsdPt"],events["n", "CryCluGenMatchAllMaxAbsdPt"] = count_idx_dpt(
        events.CryCluGenMatchAll.GenEle.idx,
        events.CryCluGenMatchAll.dPt,
        events.GenEle.pt
    )

    #!-------------------Tk-Gen Matching-------------------!#
    events["TkGenMatchAll"] = match_to_gen(events.Tk, events.GenEle)
    events["TkGenMatch"] = select_match(events.TkGenMatchAll, events.TkGenMatchAll.GenEle.idx)
    events["n", "TkGenMatchAll"], events["n", "TkGenMatchAllPt"], events["n", "TkGenMatchAllNisReal"] = count_idx_pt_isReal(
        events.TkGenMatchAll.GenEle.idx, events.GenEle.pt,events.TkGenMatchAll.Tk.isReal
    )

    _,events["n", "TkGenMatchAllMinAbsdPt"], events["n", "TkGenMatchAllMaxAbsdPt"] = count_idx_dpt(
        events.TkGenMatchAll.GenEle.idx,
        events.TkGenMatchAll.dPt,
        events.GenEle.pt,
    )

    #!-------------------TkEle-Gen Matching-------------------!#
    events["TkEleGenMatchAll"] = match_to_gen(events.TkEle, events.GenEle, etaphi_vars=(("eta", "phi"), ("caloeta", "calophi")))
    events["TkEleGenMatch"] = select_match(events.TkEleGenMatchAll, events.TkEleGenMatchAll.GenEle.idx)
    events["n", "TkEleGenMatchAll"], events["n", "TkEleGenMatchAllPt"] = count_idx_pt(
        events.TkEleGenMatchAll.GenEle.idx, events.GenEle.pt
    )
    _, events["n", "TkEleGenMatchAllMinAbsdPt"], events["n", "TkEleGenMatchAllMaxAbsdPt"] = count_idx_dpt(
        events.TkEleGenMatchAll.GenEle.idx,
        events.TkEleGenMatchAll.dPt,
        events.GenEle.pt,
    )

    #!-------------------Tk-CryClu-Gen Matching-------------------!#

    events["TkCryCluGenMatchAll"] = match_obj_to_couple(
        events.Tk, events.CryCluGenMatchAll, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
    )

    events["TkCryCluGenMatchAll","dR"]=events.TkCryCluGenMatchAll.Tk.deltaR(events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle)

    events["TkCryCluGenMatchAll","dPt"]=events.TkCryCluGenMatchAll.Tk.pt-events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.pt

    events["TkCryCluGenMatch"] = select_match(
        events.TkCryCluGenMatchAll, events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx, strategy="minabs_dPtCryClu"
    )

    events["n", "TkCryCluGenMatchAll"], events["n", "TkCryCluGenMatchAllPt"], events["n", "TkCryCluGenMatchAllNisReal"] = count_idx_pt_isReal(
        events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.GenEle.pt,
        events.TkCryCluGenMatchAll.Tk.isReal
    )

    _, events["n", "TkCryCluGenMatchAllMinAbsdPt"], events["n", "TkCryCluGenMatchAllMaxAbsdPt"] = count_idx_dpt(
        events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.TkCryCluGenMatchAll.dPt,
        events.GenEle.pt,
    )

    #!-------------------Tk-Gen+CryClu-Gen Matching-------------------!#


    events["TkGenCryCluGenMatchAll"]=events["TkCryCluGenMatchAll"][events["TkCryCluGenMatchAll","dR"]<0.1]

    events["TkGenCryCluGenMatch"] = select_match(
        events.TkGenCryCluGenMatchAll, events.TkGenCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx, strategy="minabs_dPtCryClu"
    )

    events["n", "TkGenCryCluGenMatchAll"], events["n", "TkGenCryCluGenMatchAllPt"], events["n", "TkGenCryCluGenMatchAllNisReal"] = count_idx_pt_isReal(
        events.TkGenCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.GenEle.pt,
        events.TkGenCryCluGenMatchAll.Tk.isReal,
    )

    _, events["n", "TkGenCryCluGenMatchAllMinAbsdPt"], events["n", "TkGenCryCluGenMatchAllMaxAbsdPt"] = count_idx_dpt(
        events.TkGenCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.TkGenCryCluGenMatchAll.dPt,
        events.GenEle.pt,
    )

    return events


#!In order 2D, 1D, general (due to delete on add_hist)
hists_CryCluGenMatch = [
    Hist("CryCluGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("CryCluGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("n", "CryCluGenMatchAllPt", "n", "CryCluGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "CryCluGenMatchAllMaxAbsdPt", "n", "CryCluGenMatchAll", hist_range=[(-1, 100), (0, 10)], bins=[101, 10]),
    Hist("n", "CryCluGenMatchAllMinAbsdPt", "n", "CryCluGenMatchAll", hist_range=[(-1, 100), (0, 10)], bins=[101, 10]),
]

hists_TkGenMatch = [
    Hist("TkGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatch", "dPt","TkGenMatch/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenMatch", "dR","TkGenMatch/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("TkGenMatchAll", "dPt","TkGenMatchAll/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenMatchAll", "dR","TkGenMatchAll/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("n", "TkGenMatchAllPt", "n", "TkGenMatchAll", hist_range=[(0, 100), (0, 11)], bins=[50, 11]),
    Hist("n", "TkGenMatchAllMaxAbsdPt", "n", "TkGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]),
    Hist("n", "TkGenMatchAllMinAbsdPt", "n", "TkGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]),
    Hist("n", "TkGenMatchAll", "n", "TkGenMatchAllNisReal", hist_range=[(0,11), (0, 11)], bins=[11, 11]),
]

hists_TkEleGenMatch = [
    Hist("TkEleGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkEleGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkEleGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("n", "TkEleGenMatchAllPt", "n", "TkEleGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkEleGenMatchAllMaxAbsdPt", "n", "TkEleGenMatchAll", hist_range=[(-1, 100), (0, 10)], bins=[101, 10]),
    Hist("n", "TkEleGenMatchAllMinAbsdPt", "n", "TkEleGenMatchAll", hist_range=[(-1, 100), (0, 10)], bins=[101, 10]),
]

hists_TkCryCluGenMatch = [
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkCryCluGenMatch", "dPt","TkCryCluGenMatch/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatch", "dR","TkCryCluGenMatch/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatch", "dPtCryClu","TkCryCluGenMatch/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatch", "dRCryClu","TkCryCluGenMatch/Tk", "isReal", hist_range=((0,0.2),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatchAll", "dPt","TkCryCluGenMatchAll/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatchAll", "dR","TkCryCluGenMatchAll/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatchAll", "dPtCryClu","TkCryCluGenMatchAll/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkCryCluGenMatchAll", "dRCryClu","TkCryCluGenMatchAll/Tk", "isReal", hist_range=((0,0.2),(0,3)), bins=[50,3]),
    Hist("n", "TkCryCluGenMatchAllPt", "n", "TkCryCluGenMatchAll", hist_range=[(0, 100), (0, 11)], bins=[50, 11]),
    Hist(
        "n", "TkCryCluGenMatchAllMaxAbsdPt", "n", "TkCryCluGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]
    ),
    Hist(
        "n", "TkCryCluGenMatchAllMinAbsdPt", "n", "TkCryCluGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]
    ),
    Hist("n", "TkCryCluGenMatchAll", "n", "TkCryCluGenMatchAllNisReal", hist_range=[(0,11), (0, 11)], bins=[11, 11]),
]

hists_TkGenCryCluGenMatch = [
    Hist("TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenCryCluGenMatch", "dPt","TkGenCryCluGenMatch/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatch", "dR","TkGenCryCluGenMatch/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatch", "dPtCryClu","TkGenCryCluGenMatch/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatch", "dRCryClu","TkGenCryCluGenMatch/Tk", "isReal", hist_range=((0,0.2),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatchAll", "dPt","TkGenCryCluGenMatchAll/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatchAll", "dR","TkGenCryCluGenMatchAll/Tk", "isReal", hist_range=((0,0.1),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatchAll", "dPtCryClu","TkGenCryCluGenMatchAll/Tk", "isReal", hist_range=((-100,100),(0,3)), bins=[50,3]),
    Hist("TkGenCryCluGenMatchAll", "dRCryClu","TkGenCryCluGenMatchAll/Tk", "isReal", hist_range=((0,0.2),(0,3)), bins=[50,3]),
    Hist("n", "TkGenCryCluGenMatchAllPt", "n", "TkGenCryCluGenMatchAll", hist_range=[(0, 100), (0, 11)], bins=[50, 11]),
    Hist(
        "n", "TkGenCryCluGenMatchAllMaxAbsdPt", "n", "TkGenCryCluGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]
    ),
    Hist(
        "n", "TkGenCryCluGenMatchAllMinAbsdPt", "n", "TkGenCryCluGenMatchAll", hist_range=[(-1, 100), (0, 11)], bins=[101, 11]
    ),
    Hist("n", "TkGenCryCluGenMatchAll", "n", "TkGenCryCluGenMatchAllNisReal", hist_range=[(0,11), (0, 11)], bins=[11, 11]),
]



hists_n = [
    Hist("n", hist_range=(0, 15), bins=15),
]

hists_genele = [
    Hist("GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
]

hists_all = [Hist("")]

hists = [

    *hists_CryCluGenMatch,
    *hists_TkGenMatch,
    *hists_TkEleGenMatch,
    *hists_TkCryCluGenMatch,
    *hists_TkGenCryCluGenMatch,
    *hists_genele,
    *hists_n,
    *hists_all
]
