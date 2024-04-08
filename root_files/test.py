# %%
import importlib
import sys

import awkward as ak
import numpy as np

sys.path.append("..")
import python.sample
from cfg.functions.matching import count_matched, match_to_gen, select_matched
from cfg.functions.utils import add_collection

importlib.reload(python.sample)
Sample = python.sample.Sample

path = "/data/pviscone/CMSSW/CMSSW_14_0_0_pre3/src/FastPUPPI/NtupleProducer/NanoNizer/NanoDoubleElectrons/*.root"
fname = "DoubleElectrons/DoubleElectrons.root"
path = "DoubleElectrons"
scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle"}

BarrelEta = 1.479

events = Sample("example", path=path, tree_name="Events", scheme_dict=scheme, nevents=1000)
events = events.events
add_collection(events, "n")
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
events["n", "TkGenMatchedAll"], events["n", "TkGenMatchedAllPt"] = count_matched(events.TkGenMatchedAll, events.GenEle)

#!-------------------TkEle-Gen Matching-------------------!#
events["TkEleGenMatchedAll"] = match_to_gen(events.TkEle, events.GenEle, calovar=True)
events["TkEleGenMatched"] = select_matched(events.TkEleGenMatchedAll)
events["n", "TkEleGenMatchedAll"], events["n", "TkEleGenMatchedAllPt"] = count_matched(
    events.TkEleGenMatchedAll, events.GenEle
)

# %%


def obj2obj_match(names, obj1_to_match, obj2_to_match, dr_cut=0.2, var=None):
    name1, name2 = names
    obj1_to_match = ak.with_name(obj1_to_match, "Momentum4D")
    obj2_to_match = ak.with_name(obj2_to_match, "Momentum4D")
    if var is not None:
        obj1_to_match["eta"] = obj1_to_match[var[0]["eta"]]
        obj1_to_match["phi"] = obj1_to_match[var[0]["phi"]]
        obj2_to_match["eta"] = obj2_to_match[var[1]["eta"]]
        obj2_to_match["phi"] = obj2_to_match[var[1]["phi"]]

    n = ak.max(ak.num(obj2_to_match, axis=1))
    obj2_to_match = ak.pad_none(obj2_to_match, n)
    for i in range(n):
        dr = obj2_to_match[:, i].deltaR(obj1_to_match)
        matched_obj = obj1_to_match[dr < dr_cut]
        for fields2 in obj2_to_match.fields:
            if "gen" in fields2:
                matched_obj[f"{fields2}"] = obj2_to_match[:, i][fields2]
            else:
                matched_obj[f"{name2}_{fields2}"] = obj2_to_match[:, i][fields2]
        matched_obj[f"{name2}Idx"] = i
        for fields1 in obj1_to_match.fields:
            idx = matched_obj.layout.content.fields.index(fields1)
            matched_obj.layout.content.fields[idx] = f"{name1}_{fields1}"
        matched_obj[f"{name1}_{name2}_dR"] = dr[dr < dr_cut]
        matched_obj[f"{name1}_{name2}_dPt"] = matched_obj[f"{name1}_pt"] - obj2_to_match[:, i].pt
        matched_obj["pt"] = matched_obj[f"{name2}_pt"]
        matched_obj["eta"] = matched_obj[f"{name2}_eta"]
        matched_obj["phi"] = matched_obj[f"{name2}_phi"]
        if i == 0:
            matched_objs = matched_obj
        elif i > 0:
            matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    return matched_objs


events["TkCryCluMatch"] = obj2obj_match(
    ["Tk", "CryCluGenMatchedAll"],
    events.Tk,
    events.CryCluGenMatchedAll,
    var=[{"eta": "caloEta", "phi": "caloPhi"}, {"eta": "eta", "phi": "phi"}],
)
