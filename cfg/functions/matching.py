import awkward as ak
import numba as nb
import numpy as np

from cfg.functions.utils import builders, cartesian


def match_to_gen(obj_to_match, gen, dr_cut=0.1, calovar=False):
    gen_to_match = gen
    if calovar:
        gen_to_match["old_eta"] = gen_to_match.eta
        gen_to_match["old_phi"] = gen_to_match.phi
        gen_to_match["eta"] = gen_to_match.caloeta
        gen_to_match["phi"] = gen_to_match.calophi

    cart, name1, name2 = cartesian(obj_to_match, gen_to_match)

    dr = cart[name1].deltaR(cart[name2])
    cart = cart[dr < dr_cut]
    cart["dR"] = dr[dr < dr_cut]
    cart["dPt"] = cart[name1].pt - cart[name2].pt
    if calovar:
        cart[name2, "eta"] = cart[name2, "old_eta"]
        cart[name2, "phi"] = cart[name2, "old_phi"]
        del cart[name2, "old_eta"]
        del cart[name2, "old_phi"]

    return cart


def select_match(matched_objs, idxs, strategy="min_dPt"):
    max_idx = ak.max(idxs)
    selected_list = []
    for i in range(max_idx + 1):
        mask = idxs == i

        if strategy == "min_dPt":
            selected_mask = ak.argmin(matched_objs[mask].dPt, axis=1, keepdims=True)

        if strategy == "min_dR":
            selected_mask = ak.argmin(matched_objs[mask].dR, axis=1, keepdims=True)

        if "max_pt" in strategy:
            name = strategy.split("_")[-1]
            selected_mask = ak.argmax(matched_objs[mask][name].pt, axis=1, keepdims=True)

        selected_list.append(matched_objs[mask][selected_mask])
    return ak.concatenate(selected_list, axis=1)


@builders
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


@builders(2)
@nb.jit
def count_idx_pt(builder_idx, builder_pt, idx, pt):
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
    return builder_idx, builder_pt


def match_obj_to_couple(obj, couple, to_compare, dr_cut=0.2, etaphi_vars=(("eta", "phi"), ("eta", "phi"))):
    couple_to_match = couple
    obj_to_match = obj

    couple_to_match[to_compare, "old_eta"] = couple_to_match[to_compare, "eta"]
    couple_to_match[to_compare, "old_phi"] = couple_to_match[to_compare, "phi"]
    couple_to_match[to_compare, "eta"] = couple_to_match[to_compare, etaphi_vars[1][0]]
    couple_to_match[to_compare, "phi"] = couple_to_match[to_compare, etaphi_vars[1][1]]
    obj_to_match["old_eta"] = obj_to_match["eta"]
    obj_to_match["old_phi"] = obj_to_match["phi"]
    obj_to_match["eta"] = obj_to_match[etaphi_vars[0][0]]
    obj_to_match["phi"] = obj_to_match[etaphi_vars[0][1]]

    cart, name1, name2 = cartesian(obj_to_match, couple_to_match)

    dr = cart[name1].deltaR(cart[name2, to_compare])
    cart = cart[dr < dr_cut]
    cart["dR"] = dr[dr < dr_cut]
    cart["dPt"] = cart[name1].pt - cart[name2, to_compare].pt

    cart[name1, "eta"] = cart[name1, "old_eta"]
    cart[name1, "phi"] = cart[name1, "old_phi"]

    cart[name2, to_compare, "eta"] = cart[name2, to_compare, "old_eta"]
    cart[name2, to_compare, "phi"] = cart[name2, to_compare, "old_phi"]
    del cart[name1, "old_eta"]
    del cart[name1, "old_phi"]
    del cart[name2, to_compare, "old_eta"]
    del cart[name2, to_compare, "old_phi"]
    return cart
