import awkward as ak
import dask_awkward as dak
import numpy as np

from python.hist_struct import Hist

BarrelEta = 1.479


def obj2obj_match(obj1, obj2, dr_cut=0.2, var=None):
    obj1_to_match = obj1
    obj2_to_match = obj2
    name1 = obj1.name.split("-")[0]
    name2 = obj2.name.split("-")[0]
    obj1_to_match = obj1_to_match.compute()
    obj2_to_match = obj2_to_match.compute()
    if var is not None:
        obj1_to_match["eta"] = obj1_to_match[var[0]["eta"]]
        obj1_to_match["phi"] = obj1_to_match[var[0]["phi"]]
        obj2_to_match["eta"] = obj2_to_match[var[1]["eta"]]
        obj2_to_match["phi"] = obj2_to_match[var[1]["phi"]]

    obj1_to_match = ak.with_name(obj1_to_match, "PtEtaPhiMLorentzVector")
    obj2_to_match = ak.with_name(obj2_to_match, "PtEtaPhiMLorentzVector")
    n = ak.max(ak.num(obj2_to_match, axis=1))
    obj2_to_match = ak.pad_none(obj2_to_match, n)
    for i in range(n):
        dr = obj2_to_match[:, i].delta_r(obj1_to_match)
        matched_obj = obj1_to_match[dr < dr_cut]
        argmax = ak.argmax(matched_obj.pt, axis=1, keepdims=True)
        matched_obj = matched_obj[argmax]

        for fields2 in obj2_to_match.fields:
            matched_obj[f"{name2}_{fields2}"] = ak.singletons(obj2_to_match[:, i][fields2])
        for fields1 in obj1_to_match.fields:
            idx = matched_obj.layout.content.fields.index(fields1)
            matched_obj.layout.content.fields[idx] = f"{name1}_{fields1}"
        matched_obj[f"{name1}_{name2}_dR"] = dr[dr < dr_cut][argmax]
        matched_obj[f"{name1}_{name2}_dPt"] = matched_obj[f"{name1}_pt"] - obj2_to_match[:, i].pt
        matched_obj["pt"] = matched_obj[f"{name2}_pt"]
        matched_obj["eta"] = matched_obj[f"{name2}_eta"]
        matched_obj["phi"] = matched_obj[f"{name2}_phi"]
        if i == 0:
            matched_objs = matched_obj
        elif i > 0:
            matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    return dak.from_awkward(matched_objs, 1)


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
        argmax = ak.argmax(matched_obj.pt, axis=1, keepdims=True)
        matched_obj = matched_obj[argmax]
        matched_obj["genPt"] = gen_to_match[:, i].pt
        matched_obj["genEta"] = gen_to_match[:, i].eta
        matched_obj["genPhi"] = gen_to_match[:, i].phi
        matched_obj["genIdx"] = i
        matched_obj["dPt"] = matched_obj.pt - gen_to_match[:, i].pt
        matched_obj["dR"] = dr[dr < dr_cut][argmax]
        if i == 0:
            matched_objs = matched_obj
        elif i > 0:
            matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    return dak.from_awkward(matched_objs, 1)


def define(sample):
    sample["GenEle"] = sample.GenEle[np.abs(sample.GenEle.eta) < BarrelEta]
    sample["GenEle"] = sample.GenEle[sample.GenEle.pt > 5]
    sample["CryCluGenMatched"] = gen_match(sample.CryClu, sample.GenEle, calovar=True)
    sample["TkGenMatched"] = gen_match(sample.Tk, sample.GenEle)
    sample["TkEleGenMatched"] = gen_match(sample.TkEle, sample.GenEle, calovar=True)
    sample["TkCryCluMatch"] = obj2obj_match(
        sample.Tk, sample.CryClu, var=[{"eta": "caloEta", "phi": "caloPhi"}, {"eta": "eta", "phi": "phi"}]
    )
    sample["TkCryCluGenMatch"] = gen_match(sample.TkCryCluMatch, sample.GenEle, calovar=True)
    return sample


#!WARNING: Not specifing the bins and the range force the dask.compute() before the creation of the histogram
hists = [
    Hist("TkCryCluGenMatch/genPt", hist_range=(0, 100), bins=50),
    Hist("TkCryCluGenMatch/genEta", hist_range=(-2, 2), bins=50),
    Hist("TkEleGenMatched/genPt", hist_range=(0, 100), bins=50),
    Hist("TkEleGenMatched/genEta", hist_range=(-2, 2), bins=50),
    # GenEle
    Hist("GenEle/pt", bins=50, hist_range=(0, 100)),
    Hist("GenEle/eta", bins=50, hist_range=(-2, 2)),
    Hist("GenEle/phi", hist_range=(-3.14, 3.14), bins=50),
    Hist("GenEle/caloeta", hist_range=(-2, 2), bins=50),
    Hist("GenEle/calophi", hist_range=(-3.14, 3.14), bins=50),
    # CryCluGenMatched
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
    # TkGenMatched
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
