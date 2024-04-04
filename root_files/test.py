
# %%
import importlib
import sys

import awkward as ak
import dask_awkward as dak
import numpy as np
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

sys.path.append("..")
import python.sample

importlib.reload(python.sample)
Sample = python.sample.Sample


NanoAODSchema.warn_missing_crossrefs = False

fname = "DoubleElectrons/DoubleElectrons.root"
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
).events()
BarrelEta = 1.479
events=Sample("sample",events=events)
""" events["GenEl"] = events.GenEl[np.abs(events.GenEl.eta) < BarrelEta]
events["GenEl"] = events.GenEl[events.GenEl.pt > 5]
events=events.filter(dak.num(events.GenEl)>0) """

gen = events.GenEl
obj = events.CaloEGammaCrystalClustersGCT
dr_cut = 0.1
calovar = False





def count_match(matched_objs, gen):
    print("Count")
    obj = matched_objs.compute()
    genobj = gen.compute()
    max_idx = ak.max(obj.genIdx)
    genobj = ak.pad_none(genobj, max_idx + 1)
    for i in range(max_idx + 1):
        matched_to_i = obj[obj.genIdx == i]
        matched_to_i = ak.drop_none(matched_to_i)
        n = ak.singletons(ak.num(matched_to_i))
        pt = ak.singletons(genobj.pt[:, i])
        if i == 0:
            # extend dims
            n_matched = n
            pt_matched = pt
        else:
            n_matched = ak.concatenate([n_matched, n], axis=1)
            pt_matched = ak.concatenate([pt_matched, pt], axis=1)
    return dak.from_awkward(n_matched, 1), dak.from_awkward(pt_matched, 1)



def gen_match(obj, gen, dr_cut=0.1, calovar=False):
    gen_to_match = gen.compute()
    obj_to_match = obj.compute()
    if calovar:
        gen_to_match["eta"] = gen_to_match.caloeta
        gen_to_match["phi"] = gen_to_match.calophi
    gen_to_match = ak.with_name(gen_to_match, "PtEtaPhiMLorentzVector")
    obj_to_match = ak.with_name(obj_to_match, "PtEtaPhiMLorentzVector")

    n = ak.max(ak.num(gen_to_match, axis=1))
    gen_to_match = ak.pad_none(gen_to_match, n)
    for i in range(n):
        dr = gen_to_match[:, i].delta_r(obj_to_match)
        matched_obj = obj_to_match[dr < dr_cut]
        # argmax = ak.argmax(matched_obj.pt, axis=1, keepdims=True)
        # matched_obj = matched_obj[argmax]
        matched_obj["genPt"] = gen_to_match[:, i].pt
        matched_obj["genEta"] = gen_to_match[:, i].eta
        matched_obj["genPhi"] = gen_to_match[:, i].phi
        matched_obj["genIdx"] = i
        matched_obj["dPt"] = matched_obj.pt - gen_to_match[:, i].pt
        matched_obj["dR"] = dr[dr < dr_cut]  # [argmax]
        if i == 0:
            matched_objs = matched_obj
        elif i > 0:
            matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    return dak.from_awkward(matched_objs, 1)

sample=events
sample["GenEle"] = sample.GenEl
sample["CryClu"] = sample.CaloEGammaCrystalClustersGCT
sample.add_collection("n")
#!-------------------GEN Selection-------------------!#
sample["GenEle"] = sample.GenEle[np.abs(sample.GenEle.eta) < BarrelEta]
sample["GenEle"] = sample.GenEle[sample.GenEle.pt > 5]
sample=sample.filter(dak.num(sample.GenEle) > 0)

#!-------------------CryClu-Gen Matching-------------------!#
sample["CryCluGenMatchedAll"] = gen_match(sample.CryClu, sample.GenEle, calovar=True)
sample["n", "CryCluGenMatchedAll"], sample["n", "CryCluGenMatchedAllPt"] = count_match(
    sample.CryCluGenMatchedAll, sample.GenEle
)

#!!!!!!!!!!!!!!!TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTt
# %%

gen_to_match = gen.compute()
obj_to_match = obj.compute()
if calovar:
    gen_to_match["eta"] = gen_to_match.caloeta
    gen_to_match["phi"] = gen_to_match.calophi
gen_to_match = ak.with_name(gen_to_match, "PtEtaPhiMLorentzVector")
obj_to_match = ak.with_name(obj_to_match, "PtEtaPhiMLorentzVector")

n = ak.max(ak.num(gen_to_match, axis=1))
gen_to_match = ak.pad_none(gen_to_match, n)
for i in range(n):
    dr = gen_to_match[:, i].delta_r(obj_to_match)
    matched_obj = obj_to_match[dr < dr_cut]
    #argmax = ak.argmax(matched_obj.pt, axis=1, keepdims=True)
    #matched_obj = matched_obj[argmax]
    matched_obj["genPt"] = gen_to_match[:, i].pt
    matched_obj["genEta"] = gen_to_match[:, i].eta
    matched_obj["genPhi"] = gen_to_match[:, i].phi
    matched_obj["genIdx"] = i
    matched_obj["dPt"] = matched_obj.pt - gen_to_match[:, i].pt
    matched_obj["dR"] = dr[dr < dr_cut]#[argmax]
    if i == 0:
        matched_objs = matched_obj
    elif i > 0:
        matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    # return dak.from_awkward(matched_objs, 1)
#%%
#def count(obj):

obj = matched_objs#.compute()
genobj = gen.compute()
max_idx = ak.max(obj.genIdx)
genobj = ak.pad_none(genobj, max_idx + 1)
for i in range(max_idx + 1):
    matched_to_i = obj[obj.genIdx == i]
    matched_to_i = ak.drop_none(matched_to_i)
    n = ak.singletons(ak.num(matched_to_i))
    pt = ak.singletons(genobj.pt[:, i])
    if i == 0:
        # extend dims
        n_matched = n
        pt_matched = pt
    else:
        n_matched = ak.concatenate([n_matched, n], axis=1)
        pt_matched = ak.concatenate([pt_matched, pt], axis=1)


