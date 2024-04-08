import awkward as ak
import numba as nb
import numpy as np

from cfg.functions.utils import builder, cartesian


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


@builder
@nb.njit
def count_idx(builder, idx_arr):
    for subarr in idx_arr:
        builder.begin_list()
        subarr = np.sort(np.array(subarr))  # noqa: PLW2901
        unique = np.unique(subarr)
        for i in unique:
            builder.append(np.sum(subarr == i))
        builder.end_list()
    return builder
