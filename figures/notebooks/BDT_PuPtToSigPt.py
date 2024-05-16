#%%
import importlib
import sys

import hist
import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.append("../..")
import python.plotters

importlib.reload(python.plotters)
TH1 = python.plotters.TH1
TEfficiency = python.plotters.TEfficiency
TH2 = python.plotters.TH2
TRate= python.plotters.TRate

save=False

folder="../fig/BDT_PuPtToSigPt_131Xv3/"
filename_signal = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_PuPtToSigPt_DoubleElectrons_131Xv3.root"
filename_minbias = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_PuPtToSigPt_MinBias_131Xv3.root"
signal = uproot.open(filename_signal)
minbias=uproot.open(filename_minbias)
selected_cuts=[0,0.4,0.8]
#%%
#!------------------pt-------------------

rebin=hist.rebin(2)
h2pt=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist().integrate(2)
genpt=signal["GenEle/pt;1"].to_hist()[rebin]

bdt_cuts=h2pt.axes[0].edges
pteff = TEfficiency(name="pt_eff", linewidth=3, xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,105))
for cut in selected_cuts:
    integrated=h2pt.integrate(0,hist.loc(cut),2j)[rebin]
    pteff.add(integrated,genpt,label=f"BDT>{cut:.2f}")
pteff.add(signal["TkEleGenMatch/GenEle/pt;1"].to_hist()[rebin],genpt,label="TkEle",linestyle="--",linewidth=4,color="black")

if save:
    pteff.save(folder+"pt_eff.pdf")
#%%

#!-------------------eta-------------------!#

rebin=hist.rebin(2)
h2eta=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_eta;1"].to_hist().integrate(2)
geneta=signal["GenEle/eta;1"].to_hist()

etaeff = TEfficiency(name="eta_eff", linewidth=3, xlabel=r"Gen $\eta$")
for cut in selected_cuts:
    integrated=h2eta.integrate(0,hist.loc(cut),2j)
    etaeff.add(integrated,geneta,label=f"BDT>{cut:.2f}")
etaeff.add(signal["TkEleGenMatch/GenEle/eta;1"].to_hist(),geneta,label="TkEle",linestyle="--",color="black",linewidth=4)

if save:
    etaeff.save(folder+"eta_eff.pdf")


#%%

rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,60), ylim=(8,50000),markersize=10,linewidth=4)
#!-------------------rate-------------------!#

h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()

score_cuts=h2rate.axes[1].edges[:-1]
for cut in selected_cuts:
    rate.add(h2rate[:,hist.loc(cut)],label=f"BDT score>{cut:.2f}")

rate.add(minbias["TkEle/rate_vs_ptcut;1"].to_hist(),label="TkEle")

if save:
    rate.save(folder+"rate.pdf")
# %%


#!!EFF constant in binpt
"""
tpr 97%  0.0-5.0   GeV: thr=0.34 fpr=0.26
tpr 97%  5.0-10.0  GeV: thr=0.46 fpr=0.25
tpr 97% 10.0-20.0  GeV: thr=0.49 fpr=0.20
tpr 97% 20.0-30.0  GeV: thr=0.54 fpr=0.15
tpr 97% 30.0-50.0  GeV: thr=0.57 fpr=0.14
tpr 97% 50.0-999.0 GeV: thr=0.74 fpr=0.00

"""
pt_edges=[0,5,10,20,30,50,999]
thr=[0.34,0.46,0.49,0.54,0.57,0.74]
rebin=hist.rebin(2)
h2pt=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=signal["GenEle/pt;1"].to_hist()[rebin]

hWP=0
pteff = TEfficiency(name="pt_eff", linewidth=3, xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,105))
plt.text(-15,-0.15,"score cut:",fontsize=11,color="red")
for idx,(low,high) in enumerate(zip(pt_edges[:-1],pt_edges[1:])):
    hWP=hWP+h2pt.integrate(2,hist.loc(low),hist.loc(high)).integrate(0,hist.loc(thr[idx]),2j)[rebin]
    plt.axvline(low,linestyle="--",color="red",linewidth=1)
    plt.text(low,-0.15,f"{thr[idx]:.2f}",fontsize=11,color="red")

plt.text(low+8,-0.15,f"on online pt bins",fontsize=11,color="red")


pteff.add(h2pt.integrate(2).integrate(0)[rebin],genpt,label="BDT>0")
pteff.add((h2pt.integrate(2).integrate(0)*0.97)[rebin],genpt,label="0.97*BDT>0")
pteff.add(hWP,genpt,label="97% sig eff",linewidth=4)
pteff.add(signal["TkEleGenMatch/GenEle/pt;1"].to_hist()[rebin],genpt,label="TkEle",linestyle="--",linewidth=4,color="black")


if save:
    pteff.save(folder+"pteff_097ptbins.pdf")

#%%


#!!RATE constant in binpt

rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,60),ylim=(8,50000),markersize=10,linewidth=4)


h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()

score_cuts=h2rate.axes[1].edges[:-1]

plt.text(-6,4.8,"score cut:",fontsize=11,color="red")

edges=h2rate.axes[0].edges
h=0
for idx,(low,high) in enumerate(zip(pt_edges[:-1],pt_edges[1:])):

    bitmask=np.zeros(len(edges[:-1]))
    bitmask[np.bitwise_and(edges[:-1]>=low, edges[:-1]<high)]=1

    h=h+h2rate[:,hist.loc(thr[idx])]*bitmask
    plt.text(low,4.8,f"{thr[idx]:.2f}",fontsize=11,color="red")

rate.add(h2rate[:,0],label="BDT>0")
rate.add(h,label="97% sig eff")

rate.add(minbias["TkEle/rate_vs_ptcut;1"].to_hist(),label="TkEle")
for pt in pt_edges:
    plt.axvline(pt,linestyle="--",color="red",linewidth=1)

if save:
    rate.save(folder+"rate_097ptbins.pdf")

# %%
