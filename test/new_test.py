# %%
import importlib
import sys

import awkward as ak
import numpy as np

sys.path.append("..")
import python.sample

importlib.reload(python.sample)
Sample = python.sample.Sample

from cfg.functions.matching import count_idx_pt, match_to_gen
from cfg.functions.utils import set_name,add_collection, builders
import numba as nb
fname = "../root_files/DoubleElectrons"

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle"}

BarrelEta = 1.479

events = Sample("example", path=fname, tree_name="Events", scheme_dict=scheme, nevents=1000)
events = events.events
events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]


# %%
add_collection(events, "n")
events["CryCluGenMatchAll"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)
set_name(events.CryCluGenMatchAll, "CryCluGenMatchAll")

#%%
idx,pt=    events.CryCluGenMatchAll.GenEle.idx, events.CryCluGenMatchAll.GenEle.pt
builder_idx = ak.ArrayBuilder()
builder_pt = ak.ArrayBuilder()

for event_idx, idx_ev in enumerate(idx):

    builder_idx.begin_list()
    builder_pt.begin_list()
    idx_argsort = np.argsort(np.array(idx_ev))
    idx_ev = np.array(idx_ev)[idx_argsort]
    pt_ev = pt[event_idx]
    pt_ev = np.array(pt_ev)[idx_argsort]

    unique_dict = {}
    for i, val in np.ndenumerate(idx_ev):
        if val not in unique_dict:
            unique_dict[val] = i[0]
    unique_values = np.array(list(unique_dict.keys()))
    unique_indices = np.array(list(unique_dict.values()))
    for u_idx, u_values in zip(unique_indices, unique_values):

        builder_idx.append(np.sum(idx_ev == u_values))
        builder_pt.append(pt_ev[u_idx])
    builder_idx.end_list()
    builder_pt.end_list()
    if event_idx == 464:
        break
a=builder_idx.snapshot()
apt=builder_pt.snapshot()
#%%
geneleidx = np.array([0, 1])
coupleidx = np.array([0, 0, 0])
pt = np.array([15, 16])

for gi in geneleidx:
    print(np.sum(coupleidx == gi))

    print(pt[gi])
# %%


@builders(2)
@nb.jit
def count_idx_pt(builder_n,builder_pt, couplegenidx, genpt):
    for event_idx, genpt_ev in enumerate(genpt):
        genidx_ev=np.arange(len(genpt_ev))
        couplegenidx_ev=np.array(couplegenidx[event_idx])
        builder_n.begin_list()
        builder_pt.begin_list()
        for idx in genidx_ev:
            builder_n.append(np.sum(couplegenidx_ev == idx))
            builder_pt.append(genpt_ev[idx])
        builder_n.end_list()
        builder_pt.end_list()
    return builder_n, builder_pt

a,b = count_idx_pt(events.CryCluGenMatchAll.GenEle.idx, events.GenEle.pt)

#%%

@builders(2)
@nb.jit
def count_idx_dpt(builder_n, builder_dpt, couplegenidx, coupledpt, genpt):
    for event_idx, genpt_ev in enumerate(genpt):
        genidx_ev = np.arange(len(genpt_ev))
        coupledpt_ev = np.array(coupledpt[event_idx])
        couplegenidx_ev = np.array(couplegenidx[event_idx])
        builder_n.begin_list()
        builder_dpt.begin_list()
        for idx in genidx_ev:
            builder_n.append(np.sum(couplegenidx_ev == idx))
            if len(coupledpt_ev) == 0:
                builder_dpt.append(-1)
            else:
                builder_dpt.append(np.max(np.abs(coupledpt_ev)))
        builder_n.end_list()
        builder_dpt.end_list()
    return builder_n, builder_dpt

c,d = count_idx_dpt(events.CryCluGenMatchAll.GenEle.idx,events.CryCluGenMatchAll.dPt, events.GenEle.pt)