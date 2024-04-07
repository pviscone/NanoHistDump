import awkward as ak
import numpy as np


def match_to_gen(obj_to_match, gen_to_match, dr_cut=0.1, calovar=False):
    if calovar:
        gen_to_match["eta"] = gen_to_match.caloeta
        gen_to_match["phi"] = gen_to_match.calophi
    n = ak.max(ak.num(gen_to_match, axis=1))
    gen_to_match = ak.pad_none(gen_to_match, n)
    for i in range(n):
        dr = gen_to_match[:, i].deltaR(obj_to_match)
        matched_obj = obj_to_match[dr < dr_cut]
        matched_obj["genPt"] = gen_to_match[:, i].pt
        matched_obj["genEta"] = gen_to_match[:, i].eta
        matched_obj["genPhi"] = gen_to_match[:, i].phi
        matched_obj["genIdx"] = i
        matched_obj["dPt"] = matched_obj.pt - gen_to_match[:, i].pt
        matched_obj["dR"] = dr[dr < dr_cut]
        if i == 0:
            matched_objs = matched_obj
        elif i > 0:
            matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)
    return matched_objs


def select_matched(matched_objs):
    n_matched = ak.num(matched_objs, axis=1)
    max_matched = ak.max(n_matched)
    for i in range(max_matched):
        match_to_gen_i = matched_objs[matched_objs.genIdx == i]
        selected = match_to_gen_i[ak.argmin(np.abs(match_to_gen_i.genPt - match_to_gen_i.pt), axis=1, keepdims=True)]
        if i == 0:
            selecteds = selected
        else:
            selecteds = ak.concatenate([selecteds, selected], axis=1)
    return ak.drop_none(selecteds)



def count_matched(matched_objs, gen):
    max_idx = ak.max(matched_objs.genIdx)
    genobj = ak.pad_none(gen, max_idx + 1)
    for i in range(max_idx + 1):
        matched_to_i = matched_objs[matched_objs.genIdx == i]
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
    return n_matched, pt_matched



def obj2obj_match(names,obj1_to_match, obj2_to_match, dr_cut=0.2, var=None):
    name1,name2 = names
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
    return matched_objs
