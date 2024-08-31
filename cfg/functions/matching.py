import numpy as np

from cfg.functions.utils import cartesian


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi_over_pi = dphi > np.pi
    dphi_under_neg_pi = dphi < -np.pi
    dphi = np.where(dphi_over_pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi_under_neg_pi, dphi + 2 * np.pi, dphi)
    return dphi


def delta_r(eta1, eta2, phi1, phi2):
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(dphi**2 + deta**2)


def elliptic_match(obj1, obj2, etaphi_vars, ellipse=None):
    cart, name1, name2 = cartesian(obj1, obj2, nested=True)

    obj1_name = etaphi_vars[0][0].split("/")[:-1]
    obj2_name = etaphi_vars[1][0].split("/")[:-1]
    phi1 = cart[name1][*etaphi_vars[0][1].split("/")]
    eta1 = cart[name1][*etaphi_vars[0][0].split("/")]
    phi2 = cart[name2][*etaphi_vars[1][1].split("/")]
    eta2 = cart[name2][*etaphi_vars[1][0].split("/")]

    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2

    # if ellipse is number
    assert ellipse is not None, "ellipse must be a number or a tuple of pairs of numbers"
    if isinstance(ellipse, int | float):
        mask = (dphi**2 / ellipse**2 + deta**2 / ellipse**2) < 1

    elif isinstance(ellipse, tuple | list):
        if isinstance(ellipse[0], int | float) and isinstance(ellipse[1], int | float):
            mask = (dphi**2 / ellipse[1] ** 2 + deta**2 / ellipse[0] ** 2) < 1
        else:
            mask_arr = [
                (dphi**2 / ellipse_element[1] ** 2 + deta**2 / ellipse_element[0] ** 2) < 1
                for ellipse_element in ellipse
            ]

            mask = dphi > 666
            for elem in mask_arr:
                mask = np.bitwise_or(mask, elem)
    cart = cart[mask]
    cart["dR"] = np.sqrt(dphi[mask] ** 2 + deta[mask] ** 2)
    cart["dPt"] = cart[name1][*[*obj1_name, "pt"]] - cart[name2][*[*obj2_name, "pt"]]
    cart["PtRatio"] = cart[name1][*[*obj1_name, "pt"]] / cart[name2][*[*obj2_name, "pt"]]
    cart["dEta"] = deta[mask]
    cart["dPhi"] = dphi[mask]
    return cart
