import awkward as ak
import numba as nb
import numpy as np

from cfg.functions.utils import builders, cartesian


def match_to_gen(obj, gen, dr_cut=0.1, etaphi_vars=(("eta", "phi"), ("eta", "phi")),nested=False):
    gen_to_match = gen
    obj_to_match = obj
    if etaphi_vars[1] != ("eta", "phi"):
        gen_to_match["old_eta"] = gen_to_match["eta"]
        gen_to_match["old_phi"] = gen_to_match["phi"]
        gen_to_match["eta"] = gen_to_match[etaphi_vars[1][0]]
        gen_to_match["phi"] = gen_to_match[etaphi_vars[1][1]]
    if etaphi_vars[0] != ("eta", "phi"):
        obj_to_match["old_eta"] = obj_to_match["eta"]
        obj_to_match["old_phi"] = obj_to_match["phi"]
        obj_to_match["eta"] = obj_to_match[etaphi_vars[0][0]]
        obj_to_match["phi"] = obj_to_match[etaphi_vars[0][1]]

    cart, name1, name2 = cartesian(obj_to_match, gen_to_match,nested=nested)

    dr = cart[name1].deltaR(cart[name2])
    cart = cart[dr < dr_cut]
    cart["dR"] = dr[dr < dr_cut]
    cart["dPt"] = cart[name1].pt - cart[name2].pt

    if etaphi_vars[0] != ("eta", "phi"):
        cart[name1, "eta"] = cart[name1, "old_eta"]
        cart[name1, "phi"] = cart[name1, "old_phi"]
        del cart[name1, "old_eta"]
        del cart[name1, "old_phi"]

    if etaphi_vars[1] != ("eta", "phi"):
        cart[name2, "eta"] = cart[name2, "old_eta"]
        cart[name2, "phi"] = cart[name2, "old_phi"]
        del cart[name2, "old_eta"]
        del cart[name2, "old_phi"]

    return cart


def select_match(matched_objs, idxs, strategy="minabs_dPt"):
    max_idx = ak.max(idxs)
    selected_list = []
    for i in range(max_idx + 1):
        mask = idxs == i

        func,var=strategy.split("_")


        if func.lower()=="minabs":
            selected_mask = ak.argmin(np.abs(matched_objs[mask][var]), axis=1, keepdims=True)

        elif func.lower()=="maxabs":
            selected_mask = ak.argmax(np.abs(matched_objs[mask][var]), axis=1, keepdims=True)

        elif func.lower()=="min":
            selected_mask = ak.argmin(matched_objs[mask][var], axis=1, keepdims=True)

        elif func.lower()=="max":
            selected_mask = ak.argmax(matched_objs[mask][var], axis=1, keepdims=True)



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
def count_idx_pt(builder_n, builder_pt, couplegenidx, genpt):
    for event_idx, genpt_ev in enumerate(genpt):
        genidx_ev = np.arange(len(genpt_ev))
        couplegenidx_ev = np.array(couplegenidx[event_idx])
        builder_n.begin_list()
        builder_pt.begin_list()
        for idx in genidx_ev:
            builder_n.append(np.sum(couplegenidx_ev == idx))
            builder_pt.append(genpt_ev[idx])
        builder_n.end_list()
        builder_pt.end_list()
    return builder_n, builder_pt



@builders(3)
@nb.jit
def count_idx_pt_isReal(builder_n, builder_pt, builder_isReal, couplegenidx, genpt, isReal):
    for event_idx, genpt_ev in enumerate(genpt):
        genidx_ev = np.arange(len(genpt_ev))
        couplegenidx_ev = np.array(couplegenidx[event_idx])
        builder_n.begin_list()
        builder_pt.begin_list()
        builder_isReal.begin_list()
        for idx in genidx_ev:
            builder_n.append(np.sum(couplegenidx_ev == idx))
            builder_isReal.append(np.sum(np.bitwise_and(couplegenidx_ev == idx, np.array(isReal[event_idx])==1)))
            builder_pt.append(genpt_ev[idx])
        builder_n.end_list()
        builder_pt.end_list()
        builder_isReal.end_list()
    return builder_n, builder_pt, builder_isReal


@builders(3)
@nb.jit
def count_idx_dpt(builder_n,builder_mindpt, builder_maxdpt, couplegenidx, coupledpt, genpt):
    for event_idx, genpt_ev in enumerate(genpt):
        genidx_ev = np.arange(len(genpt_ev))
        coupledpt_ev = np.array(coupledpt[event_idx])
        couplegenidx_ev = np.array(couplegenidx[event_idx])
        builder_n.begin_list()
        builder_maxdpt.begin_list()
        builder_mindpt.begin_list()
        for idx in genidx_ev:
            builder_n.append(np.sum(couplegenidx_ev == idx))
            if np.sum(couplegenidx_ev == idx) == 0:
                builder_maxdpt.append(-1)
                builder_mindpt.append(-1)
            else:
                builder_maxdpt.append(np.max(np.abs(coupledpt_ev)))
                builder_mindpt.append(np.min(np.abs(coupledpt_ev)))
        builder_n.end_list()
        builder_mindpt.end_list()
        builder_maxdpt.end_list()
    return builder_n, builder_mindpt, builder_maxdpt


def match_obj_to_couple(obj, couple, to_compare, dr_cut=0.2, etaphi_vars=(("eta", "phi"), ("eta", "phi")),nested=False):
    couple_to_match = couple
    obj_to_match = obj

    if etaphi_vars[0] != ("eta", "phi"):
        obj_to_match["old_eta"] = obj_to_match["eta"]
        obj_to_match["old_phi"] = obj_to_match["phi"]
        obj_to_match["eta"] = obj_to_match[etaphi_vars[0][0]]
        obj_to_match["phi"] = obj_to_match[etaphi_vars[0][1]]

    if etaphi_vars[1] != ("eta", "phi"):
        couple_to_match[to_compare, "old_eta"] = couple_to_match[to_compare, "eta"]
        couple_to_match[to_compare, "old_phi"] = couple_to_match[to_compare, "phi"]
        couple_to_match[to_compare, "eta"] = couple_to_match[to_compare, etaphi_vars[1][0]]
        couple_to_match[to_compare, "phi"] = couple_to_match[to_compare, etaphi_vars[1][1]]

    cart, name2, name1 = cartesian(couple_to_match, obj_to_match,nested=nested)

    deta=cart[name1].deltaeta(cart[name2, to_compare])
    dphi=cart[name1].deltaphi(cart[name2, to_compare])
    dr=np.sqrt(deta**2+dphi**2)

    cart = cart[dr < dr_cut]
    cart[f"dR{to_compare}"] = dr[dr < dr_cut]
    cart[f"dEta{to_compare}"] = deta[dr < dr_cut]
    cart[f"dPhi{to_compare}"] = dphi[dr < dr_cut]
    cart[f"dPt{to_compare}"] = cart[name1].pt - cart[name2, to_compare].pt

    if etaphi_vars[0] != ("eta", "phi"):
        cart[name1, "eta"] = cart[name1, "old_eta"]
        cart[name1, "phi"] = cart[name1, "old_phi"]
        del cart[name1, "old_eta"]
        del cart[name1, "old_phi"]

    if etaphi_vars[1] != ("eta", "phi"):
        cart[name2, to_compare, "eta"] = cart[name2, to_compare, "old_eta"]
        cart[name2, to_compare, "phi"] = cart[name2, to_compare, "old_phi"]
        del cart[name2, to_compare, "old_eta"]
        del cart[name2, to_compare, "old_phi"]
    return cart




def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi_over_pi = dphi > np.pi
    dphi_under_neg_pi = dphi < -np.pi
    dphi = np.where(dphi_over_pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi_under_neg_pi, dphi + 2 * np.pi, dphi)
    return dphi

def elliptic_match(obj1,obj2,etaphi_vars,ellipse=None):
    cart, name1, name2 = cartesian(obj1, obj2,nested=True)

    obj1_name=etaphi_vars[0][0].split("/")[:-1]
    obj2_name=etaphi_vars[1][0].split("/")[:-1]
    phi1=cart[name1][*etaphi_vars[0][1].split("/")]
    eta1=cart[name1][*etaphi_vars[0][0].split("/")]
    phi2=cart[name2][*etaphi_vars[1][1].split("/")]
    eta2=cart[name2][*etaphi_vars[1][0].split("/")]

    dphi=delta_phi(phi1,phi2)
    deta=eta1-eta2

    #if ellipse is number
    assert ellipse is not None, "ellipse must be a number or a tuple of pairs of numbers"
    if isinstance(ellipse,int|float):
        mask=(dphi**2/ellipse**2+deta**2/ellipse**2)<1

    elif isinstance(ellipse,tuple|list):
        if isinstance(ellipse[0],int|float) and isinstance(ellipse[1],int|float):
            mask=(dphi**2/ellipse[1]**2+deta**2/ellipse[0]**2)<1
        else:
            mask_arr=[(dphi**2/ellipse_element[1]**2+deta**2/ellipse_element[0]**2)<1 for ellipse_element in ellipse]

            mask=dphi>666
            for elem in mask_arr:
                mask=np.bitwise_or(mask,elem)
    cart=cart[mask]
    cart["dR"]=np.sqrt(dphi[mask]**2+deta[mask]**2)
    cart["dPt"]=cart[name1][*(obj1_name+["pt"])]-cart[name2][*(obj2_name+["pt"])]
    cart["dEta"]=deta[mask]
    cart["dPhi"]=dphi[mask]
    return ak.drop_none(cart)




def match_obj_to_obj(obj, couple, dr_cut=0.2, etaphi_vars=(("eta", "phi"), ("eta", "phi")),nested=False):
    obj2_to_match = couple
    obj_to_match = obj

    if etaphi_vars[0] != ("eta", "phi"):
        obj_to_match["old_eta"] = obj_to_match["eta"]
        obj_to_match["old_phi"] = obj_to_match["phi"]
        obj_to_match["eta"] = obj_to_match[etaphi_vars[0][0]]
        obj_to_match["phi"] = obj_to_match[etaphi_vars[0][1]]

    if etaphi_vars[1] != ("eta", "phi"):
        obj2_to_match["old_eta"] = obj2_to_match["eta"]
        obj2_to_match["old_phi"] = obj2_to_match["phi"]
        obj2_to_match["eta"] = obj2_to_match[etaphi_vars[1][0]]
        obj2_to_match["phi"] = obj2_to_match[etaphi_vars[1][1]]

    cart, name1, name2 = cartesian(obj_to_match, obj2_to_match,nested=nested)

    deta=cart[name1].deltaeta(cart[name2])
    dphi=cart[name1].deltaphi(cart[name2])
    dr=np.sqrt(deta**2+dphi**2)

    cart = cart[dr < dr_cut]
    cart["dR"] = dr[dr < dr_cut]
    cart["dPt"] = cart[name1].pt - cart[name2].pt
    cart["dEta"] = deta[dr < dr_cut]
    cart["dPhi"] = dphi[dr < dr_cut]
    #cart=ak.drop_none(cart)
    if etaphi_vars[0] != ("eta", "phi"):
        cart[name1, "eta"] = cart[name1, "old_eta"]
        cart[name1, "phi"] = cart[name1, "old_phi"]
        del cart[name1, "old_eta"]
        del cart[name1, "old_phi"]

    if etaphi_vars[1] != ("eta", "phi"):
        cart[name2, "eta"] = cart[name2, "old_eta"]
        cart[name2, "phi"] = cart[name2, "old_phi"]
        del cart[name2, "old_eta"]
        del cart[name2, "old_phi"]
    return cart
