#%%
import pathlib
import sys

sys.path.append("../../..")

import hist
import matplotlib.patheffects as mpe
import uproot

from python.plotters import TEfficiency, TRate

classes=3
dataset="train"

tag="131Xv3"
base_path=pathlib.Path(__file__).parent.joinpath("fig/out")
sig=uproot.open(base_path.joinpath(f"light{classes}class_DoubleElectronsPU200_{dataset}_{tag}.root"))
minbias=uproot.open(base_path.joinpath(f"light{classes}class_MinBias_{dataset}_{tag}.root"))

save=True

pt_edges=[0,5,10,20,30,50,150]

thr_tkEleEff_dict={
  "3_test": [1,1,0.56,0.67,0.6,0.73],
  "3_train": [1,0.95,0.57,0.65,0.58,0.73],
  "2_test": [1,1,0.56,0.67,0.58,0.72],
  "2_train": [1,0.95,0.57,0.58,0.55,0.73],
}

thr_tkEleRate_dict={
  "3_test": [0.825,0.585,0.3,0.29,0.38,0.35],
  "3_train": [0.82,0.59,0.3,0.2,0.12,0.22],
  "2_test": [0.775,0.59,0.3,0.29,0.32,0.35],
  "2_train": [0.77,0.59,0.3,0.2,0.12,0.27],
}

thr_tkEleEff=thr_tkEleEff_dict[f"{classes}_{dataset}"]
thr_tkEleRate=thr_tkEleRate_dict[f"{classes}_{dataset}"]


#!-------------------pt-------------------!#
h2pt=sig["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=sig["GenEle/pt;1"].to_hist()

h=hist.Hist(h2pt.axes[1])


pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,100),linewidth=5,rebin=5,lumitext=f"{classes} classes, {dataset} PU200")
pteff.add(sig["CryCluGenMatch/GenEle/pt;1"].to_hist(),genpt,label="Standalone")

pteff.add(h2pt.integrate(2).integrate(0),genpt,label="100%")
pteff.add_scoreCuts(h2pt,genpt,[pt_edges,thr_tkEleEff],label=f"thr={thr_tkEleEff}")
pteff.add_scoreCuts(h2pt,genpt,[pt_edges,thr_tkEleRate],label=f"thr={thr_tkEleRate}")
outline=mpe.withStroke(linewidth=8, foreground="black")
pteff.add(sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),genpt,label="TkEle",linestyle="--",linewidth=5,zorder=-99,path_effects=[outline])

if save:
    pteff.save(f"fig/class{classes}/{dataset}_eff.pdf")





#!-------------------rate-------------------!#
rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,100),markersize=11,ylim=(1e-1,5e4),linewidth=5,lumitext=f"{classes} classes, {dataset} PU200")
h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
tkelerate=minbias["TkEle/rate_vs_ptcut;1"].to_hist()
standrate=minbias["CryClu/rate_vs_ptcut;1"].to_hist()


rate.add(standrate,label="Standalone")
rate.add_scoreCuts(h2rate,0,label="100%")
rate.add_scoreCuts(h2rate,[pt_edges,thr_tkEleEff],label=f"thr={thr_tkEleEff}")
rate.add_scoreCuts(h2rate,[pt_edges,thr_tkEleRate],label=f"thr={thr_tkEleRate}")
rate.add(tkelerate,label="TkEle")

if save:
    rate.save(f"fig/class{classes}/{dataset}_rate.pdf")

# %%
