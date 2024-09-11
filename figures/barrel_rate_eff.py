# %%
import pathlib
import sys

sys.path.append("..")

import hist
import matplotlib.patheffects as mpe
import uproot

from python.plotters import TEfficiency, TRate

tag = "131Xv3"
base_path = pathlib.Path(__file__).parent.joinpath("../out")
sig = uproot.open(base_path.joinpath(f"BDT_barrel_DoubleElectronsPU200_{tag}.root"))
minbias = uproot.open(base_path.joinpath(f"BDT_barrel_MinBias_{tag}.root"))

save = False

pt_edges = [0, 5, 10, 20, 30, 50, 150]

thr_tkEleEff_dict ={"xgb":[1, 0.95, 0.57, 0.58, 0.55, 0.73],
                    "conifer": [1, 1, -0.22, 0.15, 0. , 0.1]}

thr_tkEleRate_dict = {"xgb": [0.77, 0.59, 0.3, 0.2, 0.12, 0.27],
                      "conifer": [0.18,  0.05, -0.35, -0.5, -0.6, -0.4]}

for library in ["xgb", "conifer"]:
    thr_tkEleEff=thr_tkEleEff_dict[library]
    thr_tkEleRate=thr_tkEleRate_dict[library]
    #!-------------------pt-------------------!#
    xgbscore_genpt_cryclupt = sig[f"TkCryCluGenMatch/{library}score_vs_genpt_vs_cryclupt"].to_hist()
    genpt = sig["GenEle/pt;1"].to_hist()

    newtkele_genpt = hist.Hist(xgbscore_genpt_cryclupt.axes[1])


    pteff = TEfficiency(
        name="pt_eff",
        xlabel="Gen $p_{T}$ [GeV]",
        xlim=(-5, 100),
        rebin=5,
        lumitext=f"{library} Barrel PU200",
    )
    pteff.add(sig["CryCluGenMatch/GenEle/pt"], genpt, label="Standalone")
    pteff.add(sig["TkCryCluGenMatch/CryCluGenMatch/GenEle/pt"], genpt, label="New TkEle")

    pteff.add_scoreCuts(xgbscore_genpt_cryclupt, genpt, [pt_edges, thr_tkEleEff], label=f"thr={thr_tkEleEff}")
    pteff.add_scoreCuts(xgbscore_genpt_cryclupt, genpt, [pt_edges, thr_tkEleRate], label=f"thr={thr_tkEleRate}")

    outline = mpe.withStroke(linewidth=8, foreground="black")
    pteff.add(
        sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),
        genpt,
        label="TkEle",
        linestyle="--",
        linewidth=5,
        zorder=-99,
        path_effects=[outline],
    )

    if save:
        pteff.save(f"fig/class{classes}/{dataset}_eff.pdf")


    #!-------------------rate-------------------!#
    rate = TRate(
        name="rate_vs_pt",
        xlabel="Online $p_T$ cut [GeV]",
        ylabel="Rate [kHz]",
        xlim=(0, 100),
        ylim=(1e-1, 5e4),
        fillerr=True,
        lumitext=f"{library} Barrel PU200",
    )
    h2rate = minbias[f"TkCryCluMatch/rate_pt_vs_{library}score;1"].to_hist()
    tkelerate = minbias["TkEle/rate_vs_ptcut;1"].to_hist()
    standrate = minbias["CryClu/rate_vs_ptcut;1"].to_hist()


    rate.add(standrate, label="Standalone")

    rate.add(minbias["TkCryCluMatch/rate_vs_ptcut;1"], label="New TkEle")

    rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleEff], label=f"thr={thr_tkEleEff}")
    rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleRate], label=f"thr={thr_tkEleRate}")
    rate.add(tkelerate, label="TkEle")

    if save:
        rate.save(f"fig/class{classes}/{dataset}_rate.pdf")
# %%
