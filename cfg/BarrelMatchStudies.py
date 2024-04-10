import awkward as ak
import numpy as np

from cfg.functions.matching import count_idx_pt, match_obj_to_couple, match_to_gen, select_match
from cfg.functions.utils import add_collection, set_name
from python.hist_struct import Hist

BarrelEta = 1.479


def define(events):
    add_collection(events, "n")
    #!-------------------GEN Selection-------------------!#
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
    # events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
    events = events[ak.num(events.GenEle) > 0]

    #!-------------------CryClu-Gen Matching-------------------!#
    events["CryCluGenMatchAll"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
    set_name(events.CryCluGenMatchAll, "CryCluGenMatchAll")

    events["CryCluGenMatch"] = select_match(events.CryCluGenMatchAll, events.CryCluGenMatchAll.GenEle.idx)

    events["n", "CryCluGenMatchAll"], events["n", "CryCluGenMatchAllPt"] = count_idx_pt(
        events.CryCluGenMatchAll.GenEle.idx,
        events.CryCluGenMatchAll.GenEle.pt,
    )

    _, events["n", "CryCluGenMatchAlldPt"] = count_idx_pt(
        events.CryCluGenMatchAll.GenEle.idx,
        events.CryCluGenMatchAll.dPt,
    )

    #!-------------------Tk-Gen Matching-------------------!#
    events["TkGenMatchAll"] = match_to_gen(events.Tk, events.GenEle)
    events["TkGenMatch"] = select_match(events.TkGenMatchAll, events.TkGenMatchAll.GenEle.idx)
    events["n", "TkGenMatchAll"], events["n", "TkGenMatchAllPt"] = count_idx_pt(
        events.TkGenMatchAll.GenEle.idx, events.TkGenMatchAll.GenEle.pt
    )

    _, events["n", "TkGenMatchAlldPt"] = count_idx_pt(
        events.TkGenMatchAll.GenEle.idx, events.TkGenMatchAll.dPt
    )

    #!-------------------TkEle-Gen Matching-------------------!#
    events["TkEleGenMatchAll"] = match_to_gen(events.TkEle, events.GenEle, calovar=True)
    events["TkEleGenMatch"] = select_match(events.TkEleGenMatchAll, events.TkEleGenMatchAll.GenEle.idx)
    events["n", "TkEleGenMatchAll"], events["n", "TkEleGenMatchAllPt"] = count_idx_pt(
        events.TkEleGenMatchAll.GenEle.idx, events.TkEleGenMatchAll.GenEle.pt
    )
    _, events["n", "TkEleGenMatchAlldPt"] = count_idx_pt(
        events.TkEleGenMatchAll.GenEle.idx, events.TkEleGenMatchAll.dPt
    )

    #!-------------------Tk-CryClu-Gen Matching-------------------!#

    events["TkCryCluGenMatchAll"] = match_obj_to_couple(
        events.Tk, events.CryCluGenMatchAll, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
    )

    events["TkCryCluGenMatch"] = select_match(
        events.TkCryCluGenMatchAll, events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx, strategy="min_dPt"
    )

    events["n", "TkCryCluGenMatchAll"], events["n", "TkCryCluGenMatchAllPt"] = count_idx_pt(
        events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.pt,
    )

    _, events["n", "TkCryCluGenMatchAlldPt"] = count_idx_pt(
        events.TkCryCluGenMatchAll.CryCluGenMatchAll.GenEle.idx,
        events.TkCryCluGenMatchAll.dPt,
    )
    #!-------------------Tk-CryClu-Gen @ Vertex Matching-------------------!#
    events["CryCluGenMatchAllVertex"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
    set_name(events.CryCluGenMatchAllVertex, "CryCluGenMatchAllVertex")
    events["TkCryCluGenMatchAllVertex"] = match_obj_to_couple(
        events.Tk, events.CryCluGenMatchAllVertex, "CryClu", etaphi_vars=(("eta", "phi"), ("eta", "phi"))
    )

    events["TkCryCluGenMatchVertex"] = select_match(
        events.TkCryCluGenMatchAllVertex,
        events.TkCryCluGenMatchAllVertex.CryCluGenMatchAllVertex.GenEle.idx,
        strategy="min_dPt",
    )

    events["n", "TkCryCluGenMatchAllVertex"], events["n", "TkCryCluGenMatchAllVertexPt"] = count_idx_pt(
        events.TkCryCluGenMatchAllVertex.CryCluGenMatchAllVertex.GenEle.idx,
        events.TkCryCluGenMatchAllVertex.CryCluGenMatchAllVertex.GenEle.pt,
    )

    _, events["n", "TkCryCluGenMatchAllVertexdPt"] = count_idx_pt(
        events.TkCryCluGenMatchAllVertex.CryCluGenMatchAllVertex.GenEle.idx,
        events.TkCryCluGenMatchAllVertex.dPt,
    )

    return events


#!In order 2D, 1D, general (due to delete on add_hist)
hists_CryCluGenMatch = [
    Hist("CryCluGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("CryCluGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("CryCluGenMatch"),
]

hists_TkGenMatch = [
    Hist("TkGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatch"),
]

hists_TkEleGenMatch = [
    Hist("TkEleGenMatch/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkEleGenMatch/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkEleGenMatch/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkEleGenMatch"),
]

hists_TkCryCluGenMatch = [
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkCryCluGenMatch/CryCluGenMatchAll/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkCryCluGenMatch"),
]


hists_TkCryCluGenMatchVertex = [
    Hist("TkCryCluGenMatchVertex/CryCluGenMatchAllVertex/GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatchVertex/CryCluGenMatchAllVertex/GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("TkCryCluGenMatchVertex/CryCluGenMatchAllVertex/GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkCryCluGenMatchVertex"),
]


hists_n = [
    Hist("n", "CryCluGenMatchAllPt", "n", "CryCluGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkGenMatchAllPt", "n", "TkGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkEleGenMatchAllPt", "n", "TkEleGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n", "TkCryCluGenMatchAllPt", "n", "TkCryCluGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist(
        "n",
        "TkCryCluGenMatchAllVertexPt",
        "n",
        "TkCryCluGenMatchAllVertex",
        hist_range=[(0, 100), (0, 10)],
        bins=[50, 10],
    ),
    Hist("n", "CryCluGenMatchAlldPt", "n", "CryCluGenMatchAll", hist_range=[(-100, 100), (0, 10)], bins=[100, 10]),
    Hist("n", "TkGenMatchAlldPt", "n", "TkGenMatchAll", hist_range=[(-100, 100), (0, 10)], bins=[100, 10]),
    Hist("n", "TkEleGenMatchAlldPt", "n", "TkEleGenMatchAll", hist_range=[(-100, 100), (0, 10)], bins=[100, 10]),
    Hist("n", "TkCryCluGenMatchAlldPt", "n", "TkCryCluGenMatchAll", hist_range=[(-100, 100), (0, 10)], bins=[100, 10]),
    Hist(
        "n",
        "TkCryCluGenMatchAllVertexdPt",
        "n",
        "TkCryCluGenMatchAllVertex",
        hist_range=[(-100, 100), (0, 10)],
        bins=[100, 10],
    ),
    Hist("n", hist_range=(0, 15), bins=15),
]

hists_genele = [
    Hist("GenEle", "pt", hist_range=(0, 100), bins=50),
    Hist("GenEle", "eta", hist_range=(-2, 2), bins=50),
    Hist("GenEle", "phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("GenEle"),
]

hists = [
    *hists_CryCluGenMatch,
    *hists_TkGenMatch,
    *hists_TkEleGenMatch,
    *hists_n,
    *hists_TkCryCluGenMatch,
    *hists_TkCryCluGenMatchVertex,
    *hists_genele,
]
