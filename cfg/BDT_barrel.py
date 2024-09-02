import awkward as ak
import numpy as np

from cfg.functions.matching import elliptic_match
from cfg.functions.utils import set_name
from python.hist_struct import Hist
from python.inference import xgb_wrapper

ellipse = [[0.03, 0.3]]
BarrelEta = 1.479
model = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/Barrel/barrel2classes_131Xv3.json"


features = [
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "PtRatio",
    "nMatch",
    "abs_dEta",
    "abs_dPhi",
]

features_signal = ["CryCluGenMatch_" + feat if feat.startswith("CryClu") else feat for feat in features]

to_read = ["Tk", "CryClu", "GenEle", "TkEle"]

def define(events, sample_name):
    #!-------------------TkEle -------------------!#
    events["TkEle"] = events.TkEle[np.abs(events.TkEle.eta) < BarrelEta]
    events["TkEle", "hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle", "IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"] = events.TkEle[events.TkEle.IDTightEle]


    #!-------------------New vars------------------!#
    events["CryClu", "ss"] = events.CryClu.e2x5 / events.CryClu.e5x5
    events["CryClu", "relIso"] = events.CryClu.isolation / events.CryClu.pt

    #!------------------ MinBias ------------------!#
    if "MinBias" in sample_name:
        events = events[ak.num(events.GenEle) == 0]

        events["TkCryCluMatch"] = elliptic_match(
            events.CryClu, events.Tk, etaphi_vars=[["eta", "phi"], ["caloEta", "caloPhi"]], ellipse=ellipse
        )
        events["TkCryCluMatch", "Tk", "PtFrac"] = events.TkCryCluMatch.Tk.pt / ak.sum(
            events.TkCryCluMatch.Tk.pt, axis=2
        )

        events["TkCryCluMatch", "nMatch"] = ak.num(events.TkCryCluMatch.Tk.pt, axis=2)

        events["TkCryCluMatch", "BDTscore"] = xgb_wrapper(
            model,
            events["TkCryCluMatch"],
            features=features,
            layout_template=events.TkCryCluMatch.PtRatio.layout
        )

        maxbdt_mask = ak.argmax(events["TkCryCluMatch"].BDTscore, axis=2, keepdims=True)
        events["TkCryCluMatch"] = ak.flatten(events["TkCryCluMatch"][maxbdt_mask], axis=2)

    #!-------------------Signal-------------------!#
    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatch"] = elliptic_match(
            events.GenEle, events.CryClu, etaphi_vars=[["caloeta", "calophi"], ["eta", "phi"]], ellipse=0.1
        )
        mindpt_mask = ak.argmin(np.abs(events["CryCluGenMatch"].dPt), axis=2, keepdims=True)

        events["CryCluGenMatch"] = ak.flatten(events["CryCluGenMatch"][mindpt_mask], axis=2)

        set_name(events.CryCluGenMatch, "CryCluGenMatch")

        #!-------------------Tk-CryClu-Gen Matching-------------------!#
        events["TkCryCluGenMatch"] = elliptic_match(
            events.CryCluGenMatch,
            events.Tk,
            etaphi_vars=[["CryClu/eta", "CryClu/phi"], ["caloEta", "caloPhi"]],
            ellipse=ellipse,
        )

        events["TkCryCluGenMatch", "Tk", "PtFrac"] = events.TkCryCluGenMatch.Tk.pt / ak.sum(
            events.TkCryCluGenMatch.Tk.pt, axis=2
        )

        events["TkCryCluGenMatch", "nMatch"] = ak.num(events.TkCryCluGenMatch.Tk.pt, axis=2)

        events["TkCryCluGenMatch", "BDTscore"] = xgb_wrapper(
            model,
            events["TkCryCluGenMatch"],
            features=features_signal,
            layout_template=events.TkCryCluGenMatch.PtRatio.layout,
        )

        #!-------------------BDT selection-------------------!#
        maxbdt_mask = ak.argmax(events["TkCryCluGenMatch"].BDTscore, axis=2, keepdims=True)
        events["TkCryCluGenMatch"] = ak.flatten(events["TkCryCluGenMatch"][maxbdt_mask], axis=2)

        #!-------------------TkEle-Gen Matching-------------------!#
        events["TkEleGenMatch"] = elliptic_match(
            events.GenEle, events.TkEle, etaphi_vars=[["caloeta", "calophi"], ["caloEta", "caloPhi"]], ellipse=0.1
        )
        mindpt_mask = ak.argmin(np.abs(events["TkEleGenMatch"].dPt), axis=2, keepdims=True)
        events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask], axis=2)

    return events


pt_bins = np.linspace(0, 120, 121)
eta_bins = np.linspace(-2, 2, 50)
bdt_bins = np.linspace(0, 1.01, 102)


def get_hists(sample_name):
    hists = []
    if "MinBias" in sample_name:
        hists += [
            Hist(
                ["TkCryCluMatch/CryClu~pt", "TkCryCluMatch~BDTscore"],
                bins=[pt_bins, bdt_bins],
                fill_mode="rate_pt_vs_score",
                name="TkCryCluMatch/rate_pt_vs_score",
            ),
            Hist("TkCryCluMatch/CryClu~pt", bins=pt_bins),
            Hist("TkCryCluMatch~BDTscore", bins=bdt_bins),
            # TkEle
            Hist("TkEle~pt", bins=pt_bins, fill_mode="rate_vs_ptcut"),
            # CryClu
            Hist("CryClu~pt", bins=pt_bins, fill_mode="rate_vs_ptcut"),
        ]

    else:
        hists += [  # signal
            Hist(
                [
                    "TkCryCluGenMatch~BDTscore",
                    "TkCryCluGenMatch/CryCluGenMatch/GenEle~pt",
                    "TkCryCluGenMatch/CryCluGenMatch/CryClu~pt",
                ],
                bins=[bdt_bins, pt_bins, pt_bins],
                name="TkCryCluGenMatch/score_vs_genpt_vs_cryclupt",
            ),
            Hist(
                [
                    "TkCryCluGenMatch~BDTscore",
                    "TkCryCluGenMatch/CryCluGenMatch/GenEle~eta",
                    "TkCryCluGenMatch/CryCluGenMatch/CryClu~pt",
                ],
                bins=[bdt_bins, eta_bins, pt_bins],
                name="TkCryCluGenMatch/score_vs_geneta_vs_cryclupt",
            ),
            Hist("CryCluGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle~eta", bins=eta_bins),
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~pt", bins=pt_bins),
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~eta", bins=eta_bins),
            Hist("GenEle~pt", bins=pt_bins),
            Hist("GenEle~eta", bins=eta_bins),
            Hist("TkEleGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkEleGenMatch/GenEle~eta", bins=eta_bins),
            Hist(["TkCryCluGenMatch~BDTscore", "TkCryCluGenMatch/Tk~isReal"], bins=[bdt_bins, np.array([0, 1, 2, 3])]),
            Hist("TkCryCluGenMatch~BDTscore", bins=bdt_bins),
            Hist("TkCryCluGenMatch/Tk~isReal", bins=np.array([0, 1, 2, 3])),
        ]
    return hists
