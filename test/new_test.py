# %%
import importlib
import sys

import awkward as ak
import numpy as np

sys.path.append("..")
import python.sample

importlib.reload(python.sample)
Sample = python.sample.Sample

from cfg.functions.matching import match_to_gen, set_name

path = "/data/pviscone/CMSSW/CMSSW_14_0_0_pre3/src/FastPUPPI/NtupleProducer/NanoNizer/NanoDoubleElectrons/*.root"
fname = "DoubleElectrons/DoubleElectrons.root"
path = "DoubleElectrons"
scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle"}

BarrelEta = 1.479

events = Sample("example", path=path, tree_name="Events", scheme_dict=scheme, nevents=1000)
events = events.events
events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]

# %%

def select_matched(matched_objs, strategy="min_dPt"):
    if strategy == "min_dPt":
        return matched_objs[ak.argmin(matched_objs.dPt, axis=1, keepdims=True)]
    if strategy == "min_dR":
        return matched_objs[ak.argmin(matched_objs.dR, axis=1, keepdims=True)]
    if "max_pt" in strategy:
        name = strategy.split("_")[-1]
        return matched_objs[ak.argmax(matched_objs[name].pt, axis=1, keepdims=True)]
    return None


# %%
events["CryCluGenMatchedAll"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
set_name(events.CryCluGenMatchedAll, "CryCluGenMatchedAll")

events["CryCluGenMatched"] = select_matched(events.CryCluGenMatchedAll)


#%%
max_idx=ak.max(events["CryCluGenMatchedAll"].GenEle.idx)
selected_list=[]
for i in range(max_idx+1):
    mask=events["CryCluGenMatchedAll"].GenEle.idx==i
    dPt_i=events["CryCluGenMatchedAll"][mask].dPt
    selected_mask=ak.argmin(dPt_i,axis=1,keepdims=True)
    selected_list.append(events["CryCluGenMatchedAll"][mask][selected_mask])
selected=ak.concatenate(selected_list,axis=1)

#%%
def select_matched(matched_objs, idxs,strategy="min_dPt"):
    max_idx=ak.max(idxs)
    selected_list=[]
    for i in range(max_idx+1):
        mask = idxs == i

        if strategy == "min_dPt":
            selected_mask=ak.argmin(matched_objs[mask].dPt,axis=1,keepdims=True)

        if strategy == "min_dR":
            selected_mask = ak.argmin(matched_objs[mask].dR, axis=1, keepdims=True)

        if "max_pt" in strategy:
            name = strategy.split("_")[-1]
            selected_mask = ak.argmax(matched_objs[mask][name].pt, axis=1, keepdims=True)

        selected_list.append(matched_objs[mask][selected_mask])
    return ak.concatenate(selected_list,axis=1)
