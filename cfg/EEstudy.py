import awkward as ak
import numpy as np

from cfg.functions.matching import elliptic_match
from cfg.functions.utils import add_collection, set_name
from python.hist_struct import Hist

BarrelEta = 1.479
to_read = ["HGCalClu", "TkE", "GenEle"]


def define(events, sample_name):
    #!----------------------- Objects ---------------------!#
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) >= BarrelEta]
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < 2.4]
    events = events[ak.num(events.GenEle) > 0]
    events["HGCalClu"] = events.HGCalClu[np.abs(events["HGCalClu", "eta"]) < 2.4]

    #!------------------- Clu-Gen match -------------------!#
    events["HGCalCluGenMatch"] = elliptic_match(
        events.GenEle, events.HGCalClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")), ellipse=0.4
    )

    mindpt_mask = ak.argmin(np.abs(events["HGCalCluGenMatch"].dPt), axis=2, keepdims=True)
    events["HGCalCluGenMatchSelected"] = ak.flatten(events["HGCalCluGenMatch"][mindpt_mask], axis=2)
    set_name(events.HGCalCluGenMatchSelected, "HGCalCluGenMatch")

    #!------------------ Tk-Clu-Gen match ------------------!#
    events["HGCalCluTkGenMatch"] = elliptic_match(
        events.HGCalCluGenMatchSelected,
        events.TkE,
        etaphi_vars=(("HGCalClu/eta", "HGCalClu/phi"), ("caloEta", "caloPhi")),
        ellipse=0.5,
    )
    mindpt_mask = ak.argmin(np.abs(events["HGCalCluTkGenMatch"].dPt), axis=2, keepdims=True)
    events["HGCalCluTkGenMatchSelected"] = ak.flatten(events["HGCalCluTkGenMatch"][mindpt_mask], axis=2)
    #!--------------------------- n ------------------------!#
    add_collection(events, "n")
    events["n", "HGCalCluGenMatch"] = ak.num(events.HGCalCluGenMatch, axis=2)
    events["n", "HGCalCluTkGenMatch"] = ak.num(events.HGCalCluTkGenMatch, axis=2)
    events["HGCalCluTkGenMatchSelected", "n"] = events["n", "HGCalCluTkGenMatch"]
    events["HGCalCluGenMatchSelected", "n"] = events["n", "HGCalCluGenMatch"]
    return events


pt_bins = np.linspace(0, 101, 102)
eta_bins = np.linspace(-2, 2, 50)
score_bins = np.linspace(0, 1.01, 102)
n_bins = np.linspace(0, 10, 11)

deta_bins = np.linspace(-0.2, 0.2, 120)
dphi_bins = np.linspace(-0.45, 0.45, 120)


def get_hists(sample_name):
    return [
        #!dphi,deta
        Hist(
            ["HGCalCluTkGenMatch~dPhi", "HGCalCluTkGenMatch~dEta"],
            bins=[dphi_bins, deta_bins],
            name="HGCalCluTkGenMatch/dPhi_vs_dEta",
        ),
        Hist(
            ["HGCalCluTkGenMatchSelected~dPhi", "HGCalCluTkGenMatchSelected~dEta"],
            bins=[dphi_bins, deta_bins],
            name="HGCalCluTkGenMatchSelected/dPhi_vs_dEta",
        ),
        Hist("HGCalCluTkGenMatch~dPt", bins=pt_bins),
        Hist("HGCalCluGenMatch~dPt", bins=pt_bins),
        Hist("HGCalCluGenMatchSelected~dPt", bins=pt_bins),
        #!n
        Hist("n~HGCalCluTkGenMatch", bins=n_bins),
        Hist("n~HGCalCluGenMatch", bins=n_bins),
        #!pt
        Hist("GenEle~pt", bins=pt_bins),
        Hist("HGCalCluTkGenMatchSelected/HGCalCluGenMatch/GenEle~pt", bins=pt_bins),
        Hist("HGCalCluGenMatchSelected/GenEle~pt", bins=pt_bins),
        Hist(
            ["HGCalCluGenMatchSelected/GenEle~pt", "HGCalCluGenMatchSelected/HGCalClu~pt"],
            bins=[pt_bins, pt_bins],
            name="HGCalClu_vs_GenEle/pt",
        ),
        #!pt_vs_n
        Hist(
            ["HGCalCluGenMatchSelected/GenEle~pt", "HGCalCluGenMatchSelected~n"],
            bins=[pt_bins, n_bins],
            name="HGCalCluGenMatch/pt_vs_n",
        ),
        Hist(
            ["HGCalCluTkGenMatchSelected/HGCalCluGenMatch/GenEle~pt", "HGCalCluTkGenMatchSelected~n"],
            bins=[pt_bins, n_bins],
            name="HGCalCluTkGenMatch/pt_vs_n",
        ),
        #! scores
        Hist(
            [
                "HGCalCluGenMatchSelected/GenEle~pt",
                "HGCalCluGenMatchSelected/HGCalClu~multiClassPuIdScore",
                "HGCalCluGenMatchSelected/HGCalClu~multiClassEmIdScore",
            ],
            bins=[pt_bins, score_bins, score_bins],
            name="HGCalCluGenMatch/pt_vs_puScore_vs_emScore",
        ),
        #! scores track
        Hist(
            [
                "HGCalCluTkGenMatchSelected/HGCalCluGenMatch/GenEle~pt",
                "HGCalCluTkGenMatchSelected/HGCalCluGenMatch/HGCalClu~multiClassPuIdScore",
                "HGCalCluTkGenMatchSelected/HGCalCluGenMatch/HGCalClu~multiClassEmIdScore",
            ],
            bins=[pt_bins, score_bins, score_bins],
            name="HGCalCluTkGenMatch/pt_vs_puScore_vs_emScore",
        ),
    ]
