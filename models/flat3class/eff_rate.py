#%%
import sys

sys.path.append("../..")

import hist
import matplotlib.patheffects as mpe
import uproot

from python.plotters import TEfficiency, TRate

sig=uproot.open("outfiles/BDT_flat3class_DoubleElectronsPU200_131Xv3.root")
minbias=uproot.open("outfiles/BDT_flat3class_MinBias_131Xv3.root")

save=False
"""
pt=[0,5] GeV:tpr=0.00  fpr=0.00 thr=0.99
pt=[5,10] GeV:tpr=0.04  fpr=0.00 thr=0.96
pt=[10,20] GeV:tpr=0.89  fpr=0.11 thr=0.68
pt=[20,30] GeV:tpr=0.94  fpr=0.06 thr=0.67
pt=[30,50] GeV:tpr=0.97  fpr=0.05 thr=0.77
pt=[50,150] GeV:tpr=0.99  fpr=0.02 thr=0.92

pt=[0,5] GeV:tpr=0.27  fpr=0.01 thr=0.87
pt=[5,10] GeV:tpr=0.83  fpr=0.15 thr=0.65
pt=[10,20] GeV:tpr=0.97  fpr=0.26 thr=0.25
pt=[20,30] GeV:tpr=0.99  fpr=0.28 thr=0.15
pt=[30,50] GeV:tpr=1.00  fpr=0.31 thr=0.10
pt=[50,150] GeV:tpr=1.00  fpr=0.36 thr=0.07
"""
pt_edges=[0,5,10,20,30,50,150]
eff_tkEleEff=[0.0,0.04,0.89,0.94,0.97,0.99]
thr_tkEleEff=[0.99,0.96,0.68,0.67,0.77,0.92]
eff_tkEleRate=[0.27,0.83,0.97,0.99,1.0,1.0]
thr_tkEleRate=[0.87,0.65,0.25,0.15,0.1,0.072]
rebin=3

#!-------------------pt-------------------!#
h2pt=sig["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=sig["GenEle/pt;1"].to_hist()

h=hist.Hist(h2pt.axes[1])


pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,100),linewidth=5,rebin=3)
pteff.add(sig["CryCluGenMatch/GenEle/pt;1"].to_hist(),genpt,label="Standalone")

pteff.add(h2pt.integrate(2).integrate(0),genpt,label="100%")
pteff.add_scoreCuts(h2pt,genpt,[pt_edges,thr_tkEleEff],label=f"{eff_tkEleEff}")
pteff.add_scoreCuts(h2pt,genpt,[pt_edges,thr_tkEleRate],label=f"{eff_tkEleRate}")
outline=mpe.withStroke(linewidth=8, foreground="black")
pteff.add(sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),genpt,label="TkEle",linestyle="--",linewidth=5,zorder=-99,path_effects=[outline])

if save:
    pteff.save("fig/eff.pdf")





#!-------------------rate-------------------!#
rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,100),markersize=11,ylim=(1e-1,5e4),linewidth=5)
h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
tkelerate=minbias["TkEle/rate_vs_ptcut;1"].to_hist()
standrate=minbias["CryClu/rate_vs_ptcut;1"].to_hist()


rate.add(standrate,label="Standalone")
rate.add_scoreCuts(h2rate,0,label="100%")
rate.add_scoreCuts(h2rate,[pt_edges,thr_tkEleEff],label=f"{eff_tkEleEff}")
rate.add_scoreCuts(h2rate,[pt_edges,thr_tkEleRate],label=f"{eff_tkEleRate}")
rate.add(tkelerate,label="TkEle")

if save:
    rate.save("fig/rate.pdf")




#%%

pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,100),linewidth=5,rebin=3)
pteff.add(sig["CryCluGenMatch/GenEle/pt;1"].to_hist(),genpt,label="Standalone")

"""
pt=[0,5] GeV:tpr=0.32  fpr=0.02 thr=0.85
pt=[5,10] GeV:tpr=0.80  fpr=0.13 thr=0.70
pt=[10,20] GeV:tpr=0.95  fpr=0.21 thr=0.35
pt=[20,30] GeV:tpr=0.98  fpr=0.17 thr=0.30
pt=[30,50] GeV:tpr=0.99  fpr=0.06 thr=0.55
pt=[50,150] GeV:tpr=0.99  fpr=0.08 thr=0.85
"""


pt_edges=[0,5,10,20,30,50,150]

chosen_thrs=[0.85,0.7,0.35,0.3,0.55,0.85]
chosen_eff=[0.32,0.80,0.95,0.98,0.99,0.99]

pteff.add(h2pt.integrate(2).integrate(0),genpt,label="100%")
pteff.add_scoreCuts(h2pt,genpt,[pt_edges,chosen_thrs],label=f"{chosen_eff}",color="Black")

outline=mpe.withStroke(linewidth=8, foreground="black")
pteff.add(sig["TkEleGenMatch/GenEle/pt;1"].to_hist(),genpt,label="TkEle",linestyle="--",linewidth=5,zorder=-99,path_effects=[outline])

if save:
    pteff.save("fig/chosen_eff.pdf")


rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,100),markersize=11,ylim=(1e-1,5e4),linewidth=5)
h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
tkelerate=minbias["TkEle/rate_vs_ptcut;1"].to_hist()
standrate=minbias["CryClu/rate_vs_ptcut;1"].to_hist()


rate.add(standrate,label="Standalone")
rate.add_scoreCuts(h2rate,0,label="100%")
rate.add_scoreCuts(h2rate,[pt_edges,chosen_thrs],label=f"{chosen_eff}",color="black")
rate.add(tkelerate,label="TkEle")

if save:
    rate.save("fig/chosen_rate.pdf")