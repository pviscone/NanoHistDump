# %%
import importlib
import sys

import awkward as ak
import numpy as np

sys.path.append("..")
import python.sample
from cfg.functions.utils import add_collection, builder, get_name

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

# %%


#!-------------------CryClu-Gen Matching-------------------!#
def cartesian(obj1, obj2):
    name1 = get_name(obj1)
    name2 = get_name(obj2)

    cart = ak.cartesian([obj1, obj2])
    cart = ak.zip({name1: cart["0"], name2: cart["1"]})
    argcart = ak.argcartesian([obj1, obj2])

    cart[name1, "idx"] = argcart["0"]
    cart[name2, "idx"] = argcart["1"]
    return cart, name1, name2


def match_to_gen(obj_to_match, gen, dr_cut=0.1, calovar=False):
    gen_to_match = gen
    if calovar:
        gen_to_match["eta"] = gen_to_match.caloeta
        gen_to_match["phi"] = gen_to_match.calophi

    cart, name1, name2 = cartesian(obj_to_match, gen_to_match)

    dr = cart[name1].deltaR(cart[name2])
    cart = cart[dr < dr_cut]
    cart["dR"] = dr[dr < dr_cut]
    cart["dPt"] = cart[name1].pt - cart[name2].pt
    return cart


def select_matched(matched_objs, strategy="min_dPt"):
    if strategy == "min_dPt":
        return matched_objs[ak.argmin(matched_objs.dPt, axis=1, keepdims=True)]
    if strategy == "min_dR":
        return matched_objs[ak.argmin(matched_objs.dR, axis=1, keepdims=True)]
    if "max_pt" in strategy:
        name = strategy.split("_")[-1]
        return matched_objs[ak.argmax(matched_objs[name].pt, axis=1, keepdims=True)]
    return None


import numba as nb


@builder
@nb.njit
def count_idx(builder, idx_arr):
    for subarr in idx_arr:
        builder.begin_list()
        subarr = np.sort(np.array(subarr))
        unique = np.unique(subarr)
        for i in unique:
            builder.append(np.sum(subarr == i))
        builder.end_list()
    return builder


a = match_to_gen(events.CryClu, events.GenEle, calovar=True)
n = count_idx(a.GenEle.idx)
b = select_matched(a)

# %%
import importlib
import sys

import awkward as ak
import numpy as np

sys.path.append("..")
import python.sample
from python.hist_struct import Hist

h = Hist("")
events["new"] = match_to_gen(events.CryClu, events.GenEle, calovar=True)

def doit(arr, h):
    if h.collection_name != "":
        names = h.collection_name.split("/")
        arr = arr[*names]
    def recursive(arr, h):
        if len(arr.fields)>0:
            fields=arr.fields
            for field in fields:
                if len(arr[field].fields)>0:
                    new_h = Hist(h.collection_name+"/"+field)
                else:
                    new_h = Hist(h.collection_name,field)
                recursive(arr[field], new_h)
        else:
            new_h = Hist(h.collection_name, h.var_name)
            print(h.collection_name, h.var_name)
    return recursive(arr, h)

r = doit(events, h)