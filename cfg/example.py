import awkward as ak
import numpy as np

from cfg.functions.matching import count_idx_pt, match_obj_to_couple, match_to_gen, select_matched
from cfg.functions.utils import add_collection, set_name, get_name
from python.hist_struct import Hist

BarrelEta = 1.479


def define(events):
    add_collection(events, "n")
    #!-------------------GEN Selection-------------------!#
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
    # events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
    events = events[ak.num(events.GenEle) > 0]

    #!-------------------CryClu-Gen Matching-------------------!#
    events["CryCluGenMatchedAll"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
    set_name(events.CryCluGenMatchedAll, "CryCluGenMatchedAll")

    events["CryCluGenMatched"] = select_matched(events.CryCluGenMatchedAll)

    events["n", "CryCluGenMatchedAll"], events["n", "CryCluGenMatchedAllPt"] = count_idx_pt(
        events.CryCluGenMatchedAll.GenEle.idx,
        events.CryCluGenMatchedAll.GenEle.pt,
    )

    #!-------------------Tk-Gen Matching-------------------!#
    events["TkGenMatchedAll"] = match_to_gen(events.Tk, events.GenEle)
    events["TkGenMatched"] = select_matched(events.TkGenMatchedAll)
    events["n", "TkGenMatchedAll"], events["n", "TkGenMatchedAllPt"] = count_idx_pt(
        events.TkGenMatchedAll.GenEle.idx, events.TkGenMatchedAll.GenEle.pt
    )

    #!-------------------TkEle-Gen Matching-------------------!#
    events["TkEleGenMatchedAll"] = match_to_gen(events.TkEle, events.GenEle, calovar=True)
    events["TkEleGenMatched"] = select_matched(events.TkEleGenMatchedAll)
    events["n", "TkEleGenMatchedAll"], events["n", "TkEleGenMatchedAllPt"] = count_idx_pt(
        events.TkEleGenMatchedAll.GenEle.idx, events.TkEleGenMatchedAll.GenEle.pt
    )

    #!-------------------Tk-CryClu-Gen Matching-------------------!#

    events["TkCryCluGenMatchAll"] = match_obj_to_couple(
        events.Tk, events.CryCluGenMatchedAll, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
    )

    events["TkCryCluGenMatch"] = select_matched(events.TkCryCluGenMatchAll, strategy="min_dPt")


    events["n", "TkCryCluGenMatchAll"], events["n", "TkCryCluGenMatchAllPt"] = count_idx_pt(
        events.TkCryCluGenMatchAll.CryCluGenMatchedAll.GenEle.idx,
        events.TkCryCluGenMatchAll.CryCluGenMatchedAll.GenEle.pt,
    )

    return events


#!In order 2D, 1D, general (due to delete on add_hist)


hists_CryCluGenMatched = [
    Hist("CryCluGenMatched/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatched/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("CryCluGenMatched/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("CryCluGenMatched"),
]

hists_TkGenMatched = [
    Hist("TkGenMatched/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkGenMatched/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatched/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatched"),
]

hists_TkEleGenMatched = [
    Hist("TkEleGenMatched/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkEleGenMatched/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkEleGenMatched/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkEleGenMatched"),
]

hists_TkCryCluGenMatched = [
    Hist("TkCryCluGenMatch/CryCluGenMatchedAll/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchedAll/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchedAll/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkCryCluGenMatch"),
]


hists_n = [
    Hist("n", "CryCluGenMatchedAllPt", "n", "CryCluGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkGenMatchedAllPt", "n", "TkGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkEleGenMatchedAllPt", "n", "TkEleGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkCryCluGenMatchAllPt", "n", "TkCryCluGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", hist_range=(0, 15), bins=15),
]


hists = [*hists_CryCluGenMatched, *hists_TkGenMatched, *hists_TkEleGenMatched, *hists_n, *hists_TkCryCluGenMatched]
