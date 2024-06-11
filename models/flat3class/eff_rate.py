#%%
import sys

sys.path.append("../..")

from itertools import pairwise

import hist
import matplotlib.patheffects as mpe
import numpy as np
import uproot

from python.plotters import TEfficiency, TRate

sig=uproot.open("outfiles/BDT_flat3class_DoubleElectronsPU200_131Xv3.root")
minbias=uproot.open("outfiles/BDT_flat3class_MinBias_131Xv3.root")

"""
pt=[0,5] GeV:tpr=0.97  fpr=0.79 thr=0.19
pt=[5,10] GeV:tpr=0.97  fpr=0.62 thr=0.14
pt=[10,20] GeV:tpr=0.97  fpr=0.35 thr=0.21
pt=[20,30] GeV:tpr=0.97  fpr=0.12 thr=0.42
pt=[30,50] GeV:tpr=0.97  fpr=0.06 thr=0.73
pt=[50,150] GeV:tpr=0.97  fpr=0.00 thr=0.98

pt=[0,5] GeV:tpr=0.5  fpr=0.04 thr=0.73
pt=[5,10] GeV:tpr=0.7  fpr=0.10 thr=0.76
pt=[10,20] GeV:tpr=0.9  fpr=0.14 thr=0.59
pt=[20,30] GeV:tpr=0.95  fpr=0.06 thr=0.61
pt=[30,50] GeV:tpr=0.97  fpr=0.06 thr=0.73
pt=[50,150] GeV:tpr=0.99  fpr=0.03 thr=0.82
"""
pt_edges=[0,5,10,20,30,50,150]
thr97_list=[0.19,0.14,0.21,0.42,0.73,0.98]
eff_list=[0.5,0.7,0.9,0.95,0.97,0.99]
thr_list=[0.73,0.76,0.59,0.61,0.73,0.82]

rebin=hist.rebin(3)
#%%
#!-------------------pt-------------------!#
h2pt=sig["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=sig["GenEle/pt;1"].to_hist()[rebin]


bdt_cuts=h2pt.axes[0].edges

h=hist.Hist(h2pt.axes[1])[rebin]
h97=hist.Hist(h2pt.axes[1])[rebin]

pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,100),linewidth=5)
pteff.add(sig["CryCluGenMatch/GenEle/pt;1"].to_hist()[rebin],genpt,label="Standalone")
for _,thr,thr97,(minpt,maxpt) in zip(eff_list,thr_list,thr97_list,pairwise(pt_edges)):
    integrated=h2pt.integrate(2,hist.loc(minpt),hist.loc(maxpt))
    temp_h=integrated.integrate(0,hist.loc(thr),2j)[rebin]
    temp_h97=integrated.integrate(0,hist.loc(thr97),2j)[rebin]
    h+=temp_h
    h97+=temp_h97
    import matplotlib.pyplot as plt
    plt.axvline(minpt,color="red",linestyle="--",linewidth=1.25,zorder=-2,alpha=0.6)


pteff.add(h2pt.integrate(2).integrate(0)[rebin],genpt,label="100%")
pteff.add(h97,genpt,label="97%")
pteff.add(h,genpt,label=f"{eff_list}")

outline=mpe.withStroke(linewidth=8, foreground="black")
pteff.add(sig["TkEleGenMatch/GenEle/pt;1"].to_hist()[rebin],genpt,label="TkEle",linestyle="--",linewidth=5,zorder=-99,path_effects=[outline])
pteff.save("fig/eff.pdf")


#%%



#!-------------------rate-------------------!#
rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,100),markersize=11,ylim=(1,5e4),linewidth=5)
h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
tkelerate=minbias["TkEle/rate_vs_ptcut;1"].to_hist()
standrate=minbias["CryClu/rate_vs_ptcut;1"].to_hist()
h_rate=hist.Hist(h2rate.axes[0])
h_rate97=hist.Hist(h2rate.axes[0])
for _,thr,thr97,(minpt,maxpt) in zip(eff_list,thr_list,thr97_list,pairwise(pt_edges)):
    mask=np.ones_like(h2rate.axes[0].centers)
    idx_mask=np.bitwise_and(h2rate.axes[0].centers>minpt,h2rate.axes[0].centers<maxpt)
    mask[~idx_mask]=0
    h_rate+=h2rate[:,hist.loc(thr)]*mask
    h_rate97+=h2rate[:,hist.loc(thr97)]*mask
    plt.axvline(minpt,color="red",linestyle="--",linewidth=1,alpha=0.8,zorder=-2)


rate.add(standrate,label="Standalone")
rate.add(h2rate[:,0],label="100%")
rate.add(h_rate97,label="97%")
rate.add(h_rate,label=f"{eff_list}")
rate.add(tkelerate,label="TkEle")
rate.save("fig/rate.pdf")






