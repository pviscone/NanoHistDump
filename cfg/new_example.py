
import awkward as ak
import numpy as np

from cfg.functions.new_matching import count_idx, match_to_gen, select_matched
from cfg.functions.utils import add_collection
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
    events["CryCluGenMatched"] = select_matched(events.CryCluGenMatchedAll)

    #!CI VA MESSO IL PT
    events["n", "CryCluGenMatchedAll"] = count_idx(
        events.CryCluGenMatchedAll.CryClu.idx
    )
    return events

hists = [
    Hist(
        "CryCluGenMatchedAll/CryClu",
        "pt",
        "CryCluGenMatchedAll/GenEle",
        "pt",
        hist_range=[(0, 100), (0,100)],
        bins=[50, 50],
    ),
]
