import awkward as ak
import numpy as np
import gc
from cfg.functions.matching import count_matched, match_to_gen, obj2obj_match, select_matched
from cfg.functions.utils import add_collection
from python.hist_struct import Hist

BarrelEta = 1.479

hists_TkEleGenMatched = [
    Hist("TkEleGenMatched/genPt", hist_range=(0, 100), bins=50),
    Hist("TkEleGenMatched/genEta", hist_range=(-2, 2), bins=50),
]

hists_CryCluGenMatched = [
    Hist("CryCluGenMatched/genPt", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatched/genEta", hist_range=(-2, 2), bins=50),
    Hist("CryCluGenMatched/genPhi", hist_range=(-3.14, 3.14), bins=50),
    Hist("CryCluGenMatched/dPt", hist_range=(-50, 50), bins=50),
    Hist("CryCluGenMatched/dR", hist_range=(0, 0.4), bins=50),
    Hist("CryCluGenMatched/pt", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatched/eta", hist_range=(-2, 2), bins=50),
    Hist("CryCluGenMatched/phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("CryCluGenMatched/e2x5", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatched/e5x5", hist_range=(0, 100), bins=50),
    Hist("CryCluGenMatched/isolation", hist_range=(0, 100), bins=50),
]

hists_TkGenMatched = [
    Hist("TkGenMatched/genPt", hist_range=(0, 100), bins=50),
    Hist("TkGenMatched/genEta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatched/genPhi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatched/hitPattern", hist_range=(0, 100), bins=50),
    Hist("TkGenMatched/nStubs", hist_range=(0, 100), bins=100),
    Hist("TkGenMatched/pt", hist_range=(0, 100), bins=50),
    Hist("TkGenMatched/caloEta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatched/caloPhi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatched/eta", hist_range=(-2, 2), bins=50),
    Hist("TkGenMatched/phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("TkGenMatched/chi2Bend", hist_range=(0, 20), bins=40),
    Hist("TkGenMatched/chi2RPhi", hist_range=(0, 200), bins=100),
    Hist("TkGenMatched/chi2RZ", hist_range=(0, 20), bins=40),
]

hists_TkCryCluGenMatched = [
    Hist("TkCryCluGenMatch/genPt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatch/genEta", hist_range=(-2, 2), bins=50),
]


hists_n = [
    #!-------------------Multiplicity-------------------!#
    Hist("n", hist_range=(0, 10), bins=10),
    Hist("n/CryCluGenMatchedAllPt_vs_CryCluGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n/TkGenMatchedAllPt_vs_TkGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    Hist("n/TkEleGenMatchedAllPt_vs_TkEleGenMatchedAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
    #Hist("n/TkCryCluGenMatchAllPt_vs_TkCryCluGenMatchAll", hist_range=[(0, 100), (0, 10)], bins=[50, 10]),
]

hists_gen = [
    #!-------------------GEN-------------------!#
    Hist("GenEle/pt", hist_range=(0, 100), bins=50),
    Hist("GenEle/eta", hist_range=(-2, 2), bins=50),
    Hist("GenEle/phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("GenEle/caloeta", hist_range=(-2, 2), bins=50),
    Hist("GenEle/calophi", hist_range=(-3.14, 3.14), bins=50),
]

hists = [
    *hists_n,
    *hists_gen,
    *hists_TkGenMatched,
    *hists_TkEleGenMatched,
    *hists_CryCluGenMatched,
    # *hists_TkCryCluGenMatched,
]


#!NON SALVARE LE COLLEZIONI ALL, torna solo n
def define(events):
    add_collection(events, "n")
    #!-------------------GEN Selection-------------------!#
    events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
    # events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
    events = events[ak.num(events.GenEle) > 0]

    #!-------------------CryClu-Gen Matching-------------------!#
    events["CryCluGenMatchedAll"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
    events["CryCluGenMatched"] = select_matched(events.CryCluGenMatchedAll)
    events["n", "CryCluGenMatchedAll"], events["n", "CryCluGenMatchedAllPt"] = count_matched(
        events.CryCluGenMatchedAll, events.GenEle
    )

    #!-------------------Tk-Gen Matching-------------------!#
    events["TkGenMatchedAll"] = match_to_gen(events.Tk, events.GenEle)
    events["TkGenMatched"] = select_matched(events.TkGenMatchedAll)
    events["n", "TkGenMatchedAll"], events["n", "TkGenMatchedAllPt"] = count_matched(
        events.TkGenMatchedAll, events.GenEle
    )

    #!-------------------TkEle-Gen Matching-------------------!#
    events["TkEleGenMatchedAll"] = match_to_gen(events.TkEle, events.GenEle, calovar=True)
    events["TkEleGenMatched"] = select_matched(events.TkEleGenMatchedAll)
    events["n", "TkEleGenMatchedAll"], events["n", "TkEleGenMatchedAllPt"] = count_matched(
        events.TkEleGenMatchedAll, events.GenEle
    )


    #!-------------------Tk-CryClu-Gen Matching-------------------!#

    if False:
        events["TkCryCluMatch"] = obj2obj_match(
            ["Tk", "CryClu"],
            events.Tk,
            events.CryClu,
            var=[{"eta": "caloEta", "phi": "caloPhi"}, {"eta": "eta", "phi": "phi"}],
        )
        events["TkCryCluGenMatchAll"] = match_to_gen(
            events.TkCryCluMatch, events.GenEle, calovar=True
        )
        events["TkCryCluGenMatch"] = select_matched(events.TkCryCluGenMatchAll)
        events["n", "TkCryCluGenMatchAll"], events["n", "TkCryCluGenMatchAllPt"] = count_matched(
            events.TkCryCluGenMatchAll, events.GenEle
        )


    return events
