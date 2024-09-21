import awkward as ak
import numpy as np

from cfg.functions.matching import elliptic_match
from cfg.functions.utils import set_name
from python.hist_struct import Hist
from python.inference import xgb_wrapper

ellipse = [[0.03, 0.3]]
BarrelEta = 1.479
model = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/Barrel/barrel2classes_131Xv3.json"
conifer_model = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/Barrel/conifer_barrel2classes_131Xv3/my_prj.json"
#conifer_model=None

if conifer_model:
    import conifer
    conifer_model=conifer.model.load_model(conifer_model)
    conifer_model.compile()

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

        events["TkCryCluMatch", "BDTscore"],events["TkCryCluMatch", "ConiferScore"]= xgb_wrapper(
            model,
            events["TkCryCluMatch"],
            features=features,
            conifer_model=conifer_model,
            layout_template=events.TkCryCluMatch.PtRatio.layout
        )

        maxbdt_mask = ak.argmax(events["TkCryCluMatch"].BDTscore, axis=2, keepdims=True)
        events["TkCryCluMatch"] = ak.flatten(events["TkCryCluMatch"][maxbdt_mask], axis=2)

        #!-------------------weights-------------------!#
        sf=np.load("/afs/cern.ch/work/p/pviscone/NanoHistDump/figures/DPS/pt_sf.npy")[1]
        edges=np.load("/afs/cern.ch/work/p/pviscone/NanoHistDump/figures/DPS/pt_sf.npy")[0]
        pt=ak.drop_none(ak.ravel(events.TkCryCluMatch.CryClu.pt)).to_numpy()
        idxs=np.digitize(pt,edges,right=False)-1
        global weights
        weights=ak.Array(sf[idxs])

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

        #!-------------------Tk-Gen Matching-------------------!#
        events["TkGenMatch"] = elliptic_match(
            events.GenEle, events.Tk, ellipse=[[0.13, 0.4]]
        )
        mindpt_mask = ak.argmin(np.abs(events["TkGenMatch"].dPt), axis=2, keepdims=True)

        events["TkGenMatch"] = ak.flatten(events["TkGenMatch"][mindpt_mask], axis=2)

        #!-------------------OLD Tk-Gen Matching-------------------!#
        events["OldTk"]=events.Tk[events.Tk.pt>10]
        set_name(events.OldTk, "Tk")
        events["OldTkGenMatch"] = elliptic_match(
            events.GenEle, events.OldTk, ellipse=0.2
        )
        mindpt_mask = ak.argmin(np.abs(events["OldTkGenMatch"].dPt), axis=2, keepdims=True)

        events["OldTkGenMatch"] = ak.flatten(events["OldTkGenMatch"][mindpt_mask], axis=2)

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

        events["TkCryCluGenMatch", "BDTscore"],events["TkCryCluGenMatch", "ConiferScore"] = xgb_wrapper(
            model,
            events["TkCryCluGenMatch"],
            features=features_signal,
            conifer_model=conifer_model,
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
conifer_bins = np.linspace(-2,2.01,202)


def get_hists(sample_name):
    hists = []
    if "MinBias" in sample_name:
        hists += [
            Hist("TkCryCluMatch/CryClu~pt", bins=pt_bins),
            # TkEle
            Hist("TkEle~pt", bins=pt_bins, fill_mode="rate_vs_ptcut"),
            # CryClu
            Hist("CryClu~pt", bins=pt_bins, fill_mode="rate_vs_ptcut"),
            # new tkele Rate
            Hist("TkCryCluMatch/CryClu~pt", bins=pt_bins, fill_mode="rate_vs_ptcut"),
            Hist("TkCryCluMatch~BDTscore", bins=bdt_bins),
            # Conifer
            Hist(
                ["TkCryCluMatch/CryClu~pt", "TkCryCluMatch~ConiferScore"],
                bins=[pt_bins, conifer_bins],
                fill_mode="rate_pt_vs_score",
                name="TkCryCluMatch/rate_pt_vs_coniferscore",
            ),

            #!-------------------Features weighted-------------------!#
            #pt
            Hist("TkCryCluMatch/CryClu~pt", bins=pt_bins,name="feat_w/CryClu_pt",weight=weights),
            #[0,1]
            Hist("TkCryCluMatch/CryClu~ss", bins=np.linspace(0,1,100),name="feat_w/CryClu_ss",weight=weights),
            Hist("TkCryCluMatch/CryClu~relIso", bins=np.linspace(0,1,100),name="feat_w/CryClu_relIso",weight=weights),
            Hist("TkCryCluMatch~PtRatio", bins=np.linspace(0,1,100),name="feat_w/PtRatio",weight=weights),
            Hist("TkCryCluMatch/Tk~PtFrac", bins=np.linspace(0,1,100),name="feat_w/Tk_PtFrac",weight=weights),
            #Bits
            Hist("TkCryCluMatch/CryClu~standaloneWP", bins=np.array([0,1,2]),name="feat_w/CryClu_standaloneWP",weight=weights),
            Hist("TkCryCluMatch/CryClu~looseL1TkMatchWP", bins=np.array([0,1,2]),name="feat_w/CryClu_looseL1TkMatchWP",weight=weights),
            #eta phi
            Hist("TkCryCluMatch~dEta", name="feat_w/abs_dEta", func=lambda x: np.abs(x), bins=np.linspace(-0.05,0.05,101),weight=weights),
            Hist("TkCryCluMatch~dPhi", name="feat_w/abs_dPhi", func=lambda x: np.abs(x), bins=np.linspace(-0.5,0.5,101),weight=weights),
            Hist("TkCryCluMatch~nMatch", name="feat_w/nMatch", bins=np.linspace(0,10,11),weight=weights),
            Hist("TkCryCluMatch/Tk~chi2RPhi", bins=np.linspace(0,200,201), name="feat_w/Tk_chi2RPhi",weight=weights),


            #!-------------------Features-------------------!#
            #pt
            Hist("TkCryCluMatch/CryClu~pt", bins=pt_bins,name="feat/CryClu_pt"),
            #[0,1]
            Hist("TkCryCluMatch/CryClu~ss", bins=np.linspace(0,1,100),name="feat/CryClu_ss"),
            Hist("TkCryCluMatch/CryClu~relIso", bins=np.linspace(0,1,100),name="feat/CryClu_relIso"),
            Hist("TkCryCluMatch~PtRatio", bins=np.linspace(0,1,100),name="feat/PtRatio"),
            Hist("TkCryCluMatch/Tk~PtFrac", bins=np.linspace(0,1,100),name="feat/Tk_PtFrac"),
            #Bits
            Hist("TkCryCluMatch/CryClu~standaloneWP", bins=np.array([0,1,2]),name="feat/CryClu_standaloneWP"),
            Hist("TkCryCluMatch/CryClu~looseL1TkMatchWP", bins=np.array([0,1,2]),name="feat/CryClu_looseL1TkMatchWP"),
            #eta phi
            Hist("TkCryCluMatch~dEta", name="feat/abs_dEta", func=lambda x: np.abs(x), bins=np.linspace(-0.05,0.05,101)),
            Hist("TkCryCluMatch~dPhi", name="feat/abs_dPhi", func=lambda x: np.abs(x), bins=np.linspace(-0.5,0.5,101)),
            Hist("TkCryCluMatch~nMatch", name="feat/nMatch", bins=np.linspace(0,10,11)),
            Hist("TkCryCluMatch/Tk~chi2RPhi", bins=np.linspace(0,200,201), name="feat/Tk_chi2RPhi"),

            #!-------------------BDT-------------------!#
            Hist("TkCryCluMatch~ConiferScore", bins=conifer_bins),
            Hist("TkCryCluMatch~ConiferScore", name="TkCryCluMatch/ConiferScore01", bins=bdt_bins, func=lambda x: 1/(1+np.exp(-x))),
        ]
    # signal
    elif "PU200" in sample_name:
        hists += [
            #!-------------------Efficiencies-------------------!#
            #pt
            Hist(
                [
                    "TkCryCluGenMatch~ConiferScore",
                    "TkCryCluGenMatch/CryCluGenMatch/GenEle~pt",
                    "TkCryCluGenMatch/CryCluGenMatch/CryClu~pt",
                ],
                bins=[conifer_bins, pt_bins, pt_bins],
                name="TkCryCluGenMatch/coniferscore_vs_genpt_vs_cryclupt",
            ),
            #eta
            Hist(
                [
                    "TkCryCluGenMatch~ConiferScore",
                    "TkCryCluGenMatch/CryCluGenMatch/GenEle~eta",
                    "TkCryCluGenMatch/CryCluGenMatch/CryClu~pt",
                ],
                bins=[conifer_bins, eta_bins, pt_bins],
                name="TkCryCluGenMatch/coniferscore_vs_geneta_vs_cryclupt",
            ),
            #!-------------------Matching eff-------------------!#
            Hist("GenEle~pt", bins=pt_bins),
            Hist("GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),
            Hist("CryCluGenMatch/GenEle~pt", bins=pt_bins),
            Hist("CryCluGenMatch/GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),
            Hist("TkGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkGenMatch/GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),
            Hist("OldTkGenMatch/GenEle~pt", bins=pt_bins),
            Hist("OldTkGenMatch/GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),
            Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkCryCluGenMatch/CryCluGenMatch/GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),
            Hist("TkEleGenMatch/GenEle~pt", bins=pt_bins),
            Hist("TkEleGenMatch/GenEle~eta", bins=eta_bins, func=lambda x: np.abs(x)),

            #!-------------------Features-------------------!#
            #pt
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~pt", bins=pt_bins,name="feat/CryClu_pt"),
            #[0,1]
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~ss", bins=np.linspace(0,1,100),name="feat/CryClu_ss"),
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~relIso", bins=np.linspace(0,1,100),name="feat/CryClu_relIso"),
            Hist("TkCryCluGenMatch~PtRatio", bins=np.linspace(0,1,100),name="feat/PtRatio"),
            Hist("TkCryCluGenMatch/Tk~PtFrac", bins=np.linspace(0,1,100),name="feat/Tk_PtFrac"),
            #Bits
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~standaloneWP", bins=np.array([0,1,2]),name="feat/CryClu_standaloneWP"),
            Hist("TkCryCluGenMatch/CryCluGenMatch/CryClu~looseL1TkMatchWP", bins=np.array([0,1,2]),name="feat/CryClu_looseL1TkMatchWP"),
            #eta phi
            Hist("TkCryCluGenMatch~dEta", name="feat/abs_dEta", func=lambda x: np.abs(x), bins=np.linspace(-0.05,0.05,101)),
            Hist("TkCryCluGenMatch~dPhi", name="feat/abs_dPhi", func=lambda x: np.abs(x), bins=np.linspace(-0.5,0.5,101)),
            Hist("TkCryCluGenMatch~nMatch", name="feat/nMatch", bins=np.linspace(0,10,11)),
            Hist("TkCryCluGenMatch/Tk~chi2RPhi", bins=np.linspace(0,200,201), name="feat/Tk_chi2RPhi"),
            #!-------------------BDT-------------------!#
            Hist("TkCryCluGenMatch~BDTscore", bins=bdt_bins),
            Hist("TkCryCluGenMatch~ConiferScore", bins=conifer_bins),
            Hist("TkCryCluGenMatch~ConiferScore", name="TkCryCluGenMatch/ConiferScore01", bins=bdt_bins, func=lambda x: 1/(1+np.exp(-x))),
        ]
    elif "PU0" in sample_name:
        hists+=[
            Hist(["TkCryCluGenMatch~dPhi","TkCryCluGenMatch~dEta"],bins=[np.linspace(-0.4,0.4,101),np.linspace(-0.4,0.4,101)],name="TkCryCluGenMatch/dPhi_vs_dEta"),
        ]
    return hists
