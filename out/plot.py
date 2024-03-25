# %%  # noqa: INP001
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from hist import intervals

hep.style.use("CMS")

filename = "DoubleElectrons.root"
file = uproot.open(filename)

objects = ["CryCluGenMatched", "TkGenMatched", "TkEleGenMatched", "TkCryCluGenMatch"]


def plot_efficiency(num, den, label, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    centers = (num[1][:-1] + num[1][1:]) / 2
    err = np.nan_to_num(intervals.ratio_uncertainty(num[0], den[0], "efficiency"), 0)
    eff = np.nan_to_num(num[0] / den[0], 0)
    ax.step(centers, eff, where="mid", label=label)
    ax.errorbar(centers, eff, yerr=err, fmt="none", color=ax.lines[-1].get_color())
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Efficiency")
    hep.cms.text("Phase-2 Simulation", ax=ax)
    hep.cms.lumitext("PU 200", ax=ax)
    return ax


def plot(obj, var, ax=None):
    if "CryClu" in obj and var == "eta":
        genvar = "caloeta"
    else:
        genvar = var
    if ax is None:
        _, ax = plt.subplots()
    num = file[f"{obj}/gen{var.capitalize()}"].to_numpy()
    den = file[f"GenEle/{genvar}"].to_numpy()
    ax = plot_efficiency(num, den, obj, ax=ax)
    ax.set_xlabel(f"Gen {genvar}")
    # ax.set_title(f"{obj} {var} efficiency")


for var in ["pt", "eta"]:
    fig, ax = plt.subplots()
    for obj in objects:
        plot(obj, var, ax=ax)
    if var == "eta":
        ax.set_ylim(0, 1.4)
        ax.text(
            -1.8,
            1.3,
            "Gen pT > 5 GeV",
            fontsize=20,
        )
# %%
