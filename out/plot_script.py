
# %%  # noqa: INP001
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from hist import intervals

hep.style.use("CMS")

filename = "DoubleElectrons_131Xv3.root"
file = uproot.open(filename)

objects = ["CryCluGenMatched",
           "TkGenMatched",
           "TkEleGenMatched",
           "TkCryCluGenMatch"
           ]


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
        print(obj)
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
def plot_n(file):
    objs = ["CryCluGenMatchedAll", "TkGenMatchedAll", "TkEleGenMatchedAll","TkCryCluGenMatchAll"]
    fig, ax = plt.subplots()

    for obj in objs:
        hep.histplot(file[f"n/{obj}"],label=obj,ax=ax,linewidth=2,density=True)


    ax.grid()
    ax.legend()
    ax.set_xlabel("Number of objects")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    import matplotlib.colors as colors

    for obj in objs:
        fig2, ax2 = plt.subplots()
        hep.hist2dplot(file[f"n/{obj}Pt_vs_{obj}"], label=obj, ax=ax2, norm=colors.LogNorm(vmin=1, vmax=700))
        ax2.set_xlabel("Gen pT")
        ax2.set_ylabel("Number of objects")
        ax2.set_title(obj)


plot_n(file)
# %%
