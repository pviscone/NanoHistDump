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
sig = uproot.open(base_path.joinpath(f"BDT_endcap_DoubleElectronsPU200_{tag}_KP_HGC.root"))
minbias = uproot.open(base_path.joinpath(f"BDT_endcap_MinBias_{tag}_KP_HGC.root"))

tkele_sig=uproot.open(base_path.joinpath(f"BDT_endcap_DoubleElectronsPU200_{tag}_OLD.root"))
tkele_minbias=uproot.open(base_path.joinpath(f"BDT_endcap_MinBias_{tag}_OLD.root"))
tkele_sig=sig
tkele_minbias=minbias


save = False

pt_edges = [0, 5, 10, 20, 30, 50, 150]

thr_tkEleEff_dict ={"xgb":[0.98, 0.95, 0.78, 0.35, 0., 0.],
                    "conifer": [1, 1, -0.22, 0.15, 0. , 0.]}

thr_tkEleRate_dict = {"xgb": [0.7, 0.7, 0.75, 0.5, 0.2, 0],
                      "conifer": [0.18,  0.05, -0.35, -0.5, -0.6, -0.4]}

for library in ["xgb"]:
    thr_tkEleEff=thr_tkEleEff_dict[library]
    thr_tkEleRate=thr_tkEleRate_dict[library]
    #!-------------------pt-------------------!#
    xgbscore_genpt_hgcalclupt = sig[f"TkHGCalCluGenMatch/{library}score_vs_genpt_vs_hGCalClupt"].to_hist()
    genpt = sig["GenEle/pt;1"].to_hist()

    newtkele_genpt = hist.Hist(xgbscore_genpt_hgcalclupt.axes[1])


    pteff = TEfficiency(
        name="pt_eff", xlabel="Gen $p_{T}$ [GeV]", xlim=(-5, 100), linewidth=5, rebin=5, lumitext=f"{library} Endcap PU200"
    )
    pteff.add(sig["HGCalCluGenMatch/GenEle/pt"], genpt, label="Standalone")
    pteff.add(sig["TkHGCalCluGenMatch/HGCalCluGenMatch/GenEle/pt"], genpt, label="New TkEle")

    pteff.add_scoreCuts(xgbscore_genpt_hgcalclupt, genpt, [pt_edges, thr_tkEleEff], label=f"thr={thr_tkEleEff}")
    pteff.add_scoreCuts(xgbscore_genpt_hgcalclupt, genpt, [pt_edges, thr_tkEleRate], label=f"thr={thr_tkEleRate}")

    outline = mpe.withStroke(linewidth=8, foreground="black")
    pteff.add(
        tkele_sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),
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
        log="y",
        xlim=(0, 100),
        markersize=11,
        ylim=(1e-1, 5e4),
        linewidth=5,
        lumitext=f"{library} Endcap PU200",
    )
    h2rate = minbias[f"TkHGCalCluMatch/rate_pt_vs_{library}score;1"].to_hist()
    tkelerate = tkele_minbias["TkEle/rate_vs_ptcut;1"].to_hist()
    standrate = minbias["HGCalClu/rate_vs_ptcut;1"].to_hist()


    rate.add(standrate, label="Standalone")

    rate.add(minbias["TkHGCalCluMatch/rate_vs_ptcut;1"], label="New TkEle")

    rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleEff], label=f"thr={thr_tkEleEff}")
    rate.add_scoreCuts(h2rate, [pt_edges, thr_tkEleRate], label=f"thr={thr_tkEleRate}")
    rate.add(tkelerate, label="TkEle")

    if save:
        rate.save(f"fig/class{classes}/{dataset}_rate.pdf")
# %%
