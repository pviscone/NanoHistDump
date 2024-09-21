# %%
import sys

sys.path.append("..")

import hist
import matplotlib.patheffects as mpe
import numpy as np
import uproot

import python.plotters as mplt
from python.plotters import TH1, TH2, TEfficiency, TRate

pu0 = uproot.open("/afs/cern.ch/work/p/pviscone/NanoHistDump/outDPS/DPS_DoubleElectronsPU0_v131Xv3I.root")
sig = uproot.open("/afs/cern.ch/work/p/pviscone/NanoHistDump/outDPS/DPS_DoubleElectronsPU200_v131Xv3I.root")
minbias = uproot.open("/afs/cern.ch/work/p/pviscone/NanoHistDump/outDPS/DPS_MinBias_v131Xv3I.root")


mplt.set_style("legend.frameon", False)
mplt.set_palette(mplt.ggplot_palette)

save = True
# %%
#! ---------------- Slide 15 ---------------- !#
# Rate, eff: standalone, TkEle
rate15 = TRate(
    name="rate_vs_pt",
    xlabel="Online $p_T$ thresh. [GeV]",
    ylabel="Rate [kHz]",
    xlim=(0, 60),
    ylim=(1, 1e5),
    fillerr=False,
    linewidth=0,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.35, 0.9),
    lumitext="14 TeV, 200 PU",
    grid=False,
)
rate15.add(minbias["CryClu/rate_vs_ptcut"], label=r"Standalone $e/\gamma$")
rate15.add(minbias["TkEle/rate_vs_ptcut"], label=r"Tk-matched electron")
rate15.add_text(
    0.035,
    0.975,
    r"Minimum-Bias $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=rate15.ax.transAxes,
    weight="bold",
)
rate15.add_text(
    0.7,
    0.965,
    r"$7.5 \times 10^{34}$ cm$^{-2}$ s$^{-1}$",
    fontsize=18,
    ha="left",
    va="top",
    transform=rate15.ax.transAxes,
    weight="bold",
)
if save:
    rate15.save("DPS/slides15/rate.pdf")


pteff15 = TEfficiency(
    name="pt_eff",
    xlabel=r"$p_{T}^{\text{GEN}}$ [GeV]",
    xlim=(0, 100),
    ylim=(0, 1.15),
    rebin=2,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.9, 0.4),
    lumitext="14 TeV, 200 PU",
    grid=False,
)
pteff15.add(sig["CryCluGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label=r"Standalone $e/\gamma$")
pteff15.add(sig["TkEleGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label="Tk-matched electron")
pteff15.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=pteff15.ax.transAxes,
    weight="bold",
)
pteff15.ax.axhline(0.9, color="black", linestyle="--", linewidth=1, zorder=-99)
pteff15.ax.axhline(1, color="black", linestyle="--", linewidth=1, zorder=-99)
if save:
    pteff15.save("DPS/slides15/pteff.pdf")


etaeff15 = TEfficiency(
    name="eta_eff",
    xlabel=r"$\eta^{\text{GEN}}$",
    xlim=(0, 1.479),
    ylim=(0, 1.15),
    rebin=1,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.9, 0.4),
    lumitext="14 TeV, 200 PU",
    grid=False,
)
etaeff15.add(sig["CryCluGenMatch/GenEle/eta"], sig["GenEle/eta;1"], label=r"Standalone $e/\gamma$")
etaeff15.add(sig["TkEleGenMatch/GenEle/eta"], sig["GenEle/eta;1"], label="Tk-matched electron")
etaeff15.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=etaeff15.ax.transAxes,
    weight="bold",
)
etaeff15.ax.axhline(0.9, color="black", linestyle="--", linewidth=1, zorder=-99)
etaeff15.ax.axhline(1, color="black", linestyle="--", linewidth=1, zorder=-99)
if save:
    etaeff15.save("DPS/slides15/etaeff.pdf")
# %%
#! ---------------- Slide 16 ---------------- !#
oldmatcheff16 = TEfficiency(
    name="pt_eff",
    xlabel=r"$p_{T}^{\text{GEN}}$ [GeV]",
    xlim=(0, 100),
    ylim=(0, 1.15),
    rebin=2,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.9, 0.4),
    lumitext="14 TeV, 200 PU",
    grid=False,
)
oldmatcheff16.add(sig["CryCluGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label=r"Standalone $e/\gamma$")
oldmatcheff16.add(sig["OldTkGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label="L1 Track\n($p_T^{\\text{Tk}}>10$ GeV)")
oldmatcheff16.add(sig["TkEleGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label="Tk-matched electron\n(elliptic ID)")
oldmatcheff16.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=oldmatcheff16.ax.transAxes,
    weight="bold",
)
oldmatcheff16.ax.axhline(0.9, color="black", linestyle="--", linewidth=1, zorder=-99)
oldmatcheff16.ax.axhline(1, color="black", linestyle="--", linewidth=1, zorder=-99)
if save:
    oldmatcheff16.save("DPS/slides16/oldmatcheff.pdf")

# %%
#! ---------------- Slide 17 ---------------- !#


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

etaphi17 = TH2(
    name="etaphi17",
    xlabel=r"$\Delta \Phi$ (Cluster-Track)",
    ylabel=r"$\Delta \eta$ (Cluster-Track)",
    log="z",
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    lumitext="14 TeV, 0 PU",
)
etaphi17.add(pu0["TkCryCluGenMatch/dPhi_vs_dEta"])
etaphi17.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=etaphi17.ax.transAxes,
    weight="bold",
)
etaphi17.ax.add_patch(
    Ellipse(
        (0, 0),
        0.3 * 2,
        0.03 * 2,
        color="red",
        fill=False,
        label=r"Ellipse ($|\Delta \phi| = 0.3$, $|\Delta \eta| = 0.03$)",
    )
)
plt.legend(bbox_to_anchor=(0.98, 0.9))
if save:
    etaphi17.save("DPS/slides17/etaphi.pdf")


newmatcheff17 = TEfficiency(
    name="pt_eff",
    xlabel=r"$p_{T}^{\text{GEN}}$ [GeV]",
    xlim=(0, 100),
    ylim=(0, 1.15),
    rebin=2,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.9, 0.4),
    lumitext="14 TeV, 200 PU",
    grid=False,
)
newmatcheff17.add(sig["CryCluGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label=r"Standalone $e/\gamma$")
newmatcheff17.add(sig["TkGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label="L1 Track")
newmatcheff17.add(
    sig["TkCryCluGenMatch/CryCluGenMatch/GenEle/pt"], sig["GenEle/pt;1"], label="Tk-matched electron\n(elliptic ID)"
)
newmatcheff17.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=newmatcheff17.ax.transAxes,
    weight="bold",
)
newmatcheff17.ax.axhline(0.9, color="black", linestyle="--", linewidth=1, zorder=-99)
newmatcheff17.ax.axhline(1, color="black", linestyle="--", linewidth=1, zorder=-99)
if save:
    newmatcheff17.save("DPS/slides17/oldmatcheff.pdf")


# %%
#! ---------------- Slide 18 ---------------- !#
import matplotlib.pyplot as plt
import mplhep as hep

features = sig["feat"].keys()
labels = {
    "CryClu_pt": r"$p_T^{\text{Cluster}}$ [GeV]",
    "CryClu_ss": r"$E^{\text{Cluster}}_{2\times5}/E^{\text{Cluster}}_{5\times5}$",
    "CryClu_relIso": r"Cluster Iso./$p^{\text{Cluster}}_T$",
    "CryClu_standaloneWP": r"Cluster StandaloneWP",
    "CryClu_looseL1TkMatchWP": r"Cluster LooseL1TkMatchWP",
    "Tk_chi2RPhi": r"Tk $\chi^2_{\text{R-}\phi}$",
    "Tk_PtFrac": r"$p_T^{\text{Tk}}/\sum p_T^{\text{Matched Tk}}$",
    "abs_dEta": r"$|\Delta \eta|$ (Tk-Cluster)",
    "abs_dPhi": r"$|\Delta \phi|$ (Tk-Cluster)",
    "nMatch": r"$N_{\text{Matched Tracks}}$",
    "PtRatio": r"$p_T^{\text{Tk}}/p_T^{\text{Cluster}}$",
}
features = labels.keys()


fig, axs = plt.subplots(4, 3, figsize=(25, 20), clip_on=True)

hep.cms.text("Phase-2 Simulation Preliminary", ax=axs[0, 0], fontsize=24)
hep.cms.lumitext("14 TeV, 200 PU", ax=axs[0, 2], fontsize=24)
for idx, feat in enumerate(features):
    ax = axs[idx // 3, idx % 3]
    sig_hist = sig["feat"][feat].to_hist()
    minbias_hist = minbias["feat"][feat].to_hist()
    if "dEta" in feat:
        sig_hist = sig_hist[hist.loc(0) : hist.loc(0.03)]
        minbias_hist = minbias_hist[hist.loc(0) : hist.loc(0.03)]
    elif "dPhi" in feat:
        sig_hist = sig_hist[hist.loc(0) : hist.loc(0.3)]
        minbias_hist = minbias_hist[hist.loc(0) : hist.loc(0.3)]
    elif "chi2RPhi" in feat:
        sig_hist = sig_hist[0 : hist.loc(10)]
        minbias_hist = minbias_hist[0 : hist.loc(10)]
    elif "nMatch" in feat:
        sig_hist = sig_hist[hist.loc(1) :]
        minbias_hist = minbias_hist[hist.loc(1) :]

    if feat not in ["CryClu_standaloneWP", "CryClu_looseL1TkMatchWP", "nMatch"]:
        sig_hist = sig_hist[hist.rebin(2)]
        minbias_hist = minbias_hist[hist.rebin(2)]
    sig_hist.plot(
        ax=ax,
        label="Signal",
        color="darkorange",
        density=True,
    )
    minbias_hist.plot(
        ax=ax,
        label="Background",
        color="dodgerblue",
        density=True,
    )
    ax.set_xlabel(labels[feat])
    ax.set_ylabel("a.u.")

    if feat not in ["CryClu_standaloneWP", "CryClu_looseL1TkMatchWP"]:
        ax.set_yscale("log")

    if "CryClu" in feat:
        ax.patch.set_facecolor("lightgreen")
        ax.patch.set_alpha(0.1)
    elif "Tk_chi2RPhi" in feat:
        ax.patch.set_facecolor("lightblue")
        ax.patch.set_alpha(0.1)
    else:
        ax.patch.set_facecolor("salmon")
        ax.patch.set_alpha(0.1)

axs[-1, -1].axis("off")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

signal = Line2D([0], [0], color="darkorange", label="Signal")
background = Line2D([0], [0], color="dodgerblue", label="Background")
clu_patch = mpatches.Patch(color="lightgreen", alpha=0.1, label="ECAL cluster")
tk_patch = mpatches.Patch(color="lightblue", alpha=0.1, label="L1 track")
match_patch = mpatches.Patch(color="salmon", alpha=0.1, label="Calo-Tk matching")
leg1 = plt.legend(
    handles=[clu_patch, tk_patch, match_patch], loc="upper right", fontsize=22, frameon=True, title="Feature type"
)
plt.legend(handles=[signal, background], loc="upper left", fontsize=22)
plt.gca().add_artist(leg1)
if save:
    plt.savefig("DPS/slides18/features.pdf")


# %%
#! ---------------- Slide 19 ---------------- !#
fig, axs = plt.subplots(4, 3, figsize=(25, 20), clip_on=True)

hep.cms.text("Phase-2 Simulation Preliminary", ax=axs[0, 0], fontsize=24)
hep.cms.lumitext("14 TeV, 200 PU", ax=axs[0, 2], fontsize=24)
for idx, feat in enumerate(features):
    ax = axs[idx // 3, idx % 3]
    sig_hist = sig["feat"][feat].to_hist()
    minbias_hist = minbias["feat_w"][feat].to_hist()
    if "dEta" in feat:
        sig_hist = sig_hist[hist.loc(0) : hist.loc(0.03)]
        minbias_hist = minbias_hist[hist.loc(0) : hist.loc(0.03)]
    elif "dPhi" in feat:
        sig_hist = sig_hist[hist.loc(0) : hist.loc(0.3)]
        minbias_hist = minbias_hist[hist.loc(0) : hist.loc(0.3)]
    elif "chi2RPhi" in feat:
        sig_hist = sig_hist[0 : hist.loc(10)]
        minbias_hist = minbias_hist[0 : hist.loc(10)]
    elif "nMatch" in feat:
        sig_hist = sig_hist[hist.loc(1) :]
        minbias_hist = minbias_hist[hist.loc(1) :]

    if feat not in ["CryClu_standaloneWP", "CryClu_looseL1TkMatchWP", "nMatch"]:
        sig_hist = sig_hist[hist.rebin(2)]
        minbias_hist = minbias_hist[hist.rebin(2)]
    sig_hist.plot(
        ax=ax,
        label="Signal",
        color="darkorange",
        density=True,
    )
    minbias_hist.plot(
        ax=ax,
        label="Background",
        color="dodgerblue",
        density=True,
    )
    ax.set_xlabel(labels[feat])
    ax.set_ylabel("a.u.")

    if feat not in ["CryClu_standaloneWP", "CryClu_looseL1TkMatchWP"]:
        ax.set_yscale("log")

    if "CryClu" in feat:
        ax.patch.set_facecolor("lightgreen")
        ax.patch.set_alpha(0.1)
    elif "Tk_chi2RPhi" in feat:
        ax.patch.set_facecolor("lightblue")
        ax.patch.set_alpha(0.1)
    else:
        ax.patch.set_facecolor("salmon")
        ax.patch.set_alpha(0.1)

axs[-1, -1].axis("off")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

signal = Line2D([0], [0], color="darkorange", label="Signal")
background = Line2D([0], [0], color="dodgerblue", label="Background")
clu_patch = mpatches.Patch(color="lightgreen", alpha=0.1, label="ECAL cluster")
tk_patch = mpatches.Patch(color="lightblue", alpha=0.1, label="L1 track")
match_patch = mpatches.Patch(color="salmon", alpha=0.1, label="Calo-Tk matching")
leg1 = plt.legend(
    handles=[clu_patch, tk_patch, match_patch], loc="upper right", fontsize=22, frameon=True, title="Feature type"
)
plt.legend(handles=[signal, background], loc="upper left", fontsize=22)
plt.gca().add_artist(leg1)
axs[-1, -1].text(
    0.5,
    0.25,
    r"$p^{\text{Cluster}}_T$ flattened",
    fontsize=24,
    ha="center",
    va="center",
    transform=axs[-1, -1].transAxes,
    weight="bold",
)
if save:
    plt.savefig("DPS/slides19/features_flat.pdf")

# %%
#! ---------------- Slide 20 ---------------- !#
scores20 = TH1(
    name="etaphi17",
    xlabel="BDT Score",
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    lumitext="14 TeV, 200 PU",
    density=True,
    rebin=3,
    grid=False,
    ylabel="a.u.",
    # ylim=(1e-3, 10),
)
scores20.add(sig["TkCryCluGenMatch/ConiferScore01"], label="Signal")
scores20.add(minbias["TkCryCluMatch/ConiferScore01"], label="Background")
if save:
    scores20.save("DPS/slides20/scores.pdf")

# %%
#! ---------------- Slide 21 ---------------- !#

mplt.set_palette(mplt.acab_palette)
pt_edges = [0, 5, 10, 20, 30, 50, 150]

thr_tkEleEff_dict = {"xgb": [1, 0.95, 0.57, 0.58, 0.55, 0.73], "conifer": [8, 8, 0.05, -0.1, -0.2, 0.15]}

thr_tkEleRate_dict = {"xgb": [0.77, 0.59, 0.3, 0.2, 0.12, 0.27], "conifer": [0.19, 0.05, -0.35, -0.45, -0.5, -0.38]}

library = "conifer"
thr_tkEleEff = np.array(thr_tkEleEff_dict[library]) * 4
thr_tkEleRate = np.array(thr_tkEleRate_dict[library]) * 4


#!-------------------pt-------------------!#
xgbscore_genpt_cryclupt = sig[f"TkCryCluGenMatch/{library}score_vs_genpt_vs_cryclupt"].to_hist()
genpt = sig["GenEle/pt;1"].to_hist()

newtkele_genpt = hist.Hist(xgbscore_genpt_cryclupt.axes[1])


pteff = TEfficiency(
    name="pt_eff",
    xlabel=r"$p_{T}^{\text{GEN}}$ [GeV]",
    xlim=(0, 100),
    ylim=(0, 1.15),
    rebin=2,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.9, 0.4),
    lumitext="14 TeV, 200 PU",
    linewidth=2,
    grid=False,
    avxalpha=0,
)
pteff.add(sig["CryCluGenMatch/GenEle/pt"], genpt, label=r"Standalone $e/\gamma$")
pteff.add(sig["TkCryCluGenMatch/CryCluGenMatch/GenEle/pt"], genpt, label="Loose elliptic ID")

pteff.add_scoreCuts(xgbscore_genpt_cryclupt, genpt, [pt_edges, thr_tkEleEff], label="Composite ID TightWP")
pteff.add_scoreCuts(xgbscore_genpt_cryclupt, genpt, [pt_edges, thr_tkEleRate], label="Composite ID LooseWP")

outline = mpe.withStroke(linewidth=1, foreground="black")
pteff.add(
    sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),
    genpt,
    label="Elliptic ID",
    linestyle="--",
    linewidth=2,
    zorder=10,
    path_effects=[outline],
)
pteff.add_text(
    0.035,
    0.975,
    r"flat-$p_T$ electrons, $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=pteff.ax.transAxes,
    weight="bold",
)
pteff.ax.axhline(0.9, color="black", linestyle="--", linewidth=1, zorder=-99)
pteff.ax.axhline(1, color="black", linestyle="--", linewidth=1, zorder=-99)

if save:
    pteff.save("DPS/slides21/pteff.pdf")

# %%


#!-------------------rate-------------------!#
rate = TRate(
    name="rate_vs_pt",
    xlabel="Online $p_T$ thresh. [GeV]",
    ylabel="Rate [kHz]",
    xlim=(0, 60),
    ylim=(1, 1e5),
    fillerr=False,
    linewidth=0,
    markersize=7.5,
    cmstext="Phase-2 Simulation Preliminary",
    cmstextsize=22,
    lumitextsize=22,
    legendpos=(0.35, 0.9),
    lumitext="14 TeV, 200 PU",
    grid=False,
    avxalpha=0
)
h2rate = minbias[f"TkCryCluMatch/rate_pt_vs_{library}score;1"].to_hist()
tkelerate = minbias["TkEle/rate_vs_ptcut;1"].to_hist()
standrate = minbias["CryClu/rate_vs_ptcut;1"].to_hist()


rate.add(standrate, label=r"Standalone $e/\gamma$")

rate.add(minbias["TkCryCluMatch/rate_vs_ptcut;1"], label="Loose elliptic ID")

rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleEff], label="Composite ID TightWP")
rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleRate], label="Composite ID LooseWP")
outline = mpe.withStroke(linewidth=4, foreground="black")
rate.add(tkelerate, label="Elliptic ID")
rate.add_text(
    0.035,
    0.975,
    r"Minimum-Bias $|\eta^{\text{L1}}|<1.479$",
    fontsize=18,
    ha="left",
    va="top",
    transform=rate.ax.transAxes,
    weight="bold",
)
if save:
    rate.save("DPS/slides21/rate.pdf")

# %%
