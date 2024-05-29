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

folder="../fig/BDT_SigPtToPuPt_131Xv3/"
filename_signal = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_SigPtToPuPt_DoubleElectrons_131Xv3.root"
filename_minbias = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_SigPtToPuPt_MinBias_131Xv3.root"
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
tpr 97%  0.0-5.0   GeV: thr=0.54 fpr=0.12
tpr 97%  5.0-10.0  GeV: thr=0.52 fpr=0.21
tpr 97% 10.0-20.0  GeV: thr=0.52 fpr=0.25
tpr 97% 20.0-30.0  GeV: thr=0.52 fpr=0.34
tpr 97% 30.0-50.0  GeV: thr=0.51 fpr=0.51
tpr 97% 50.0-999.0 GeV: thr=0.50 fpr=0.64


tpr 99%  0.0-5.0   GeV: thr=0.23 fpr=0.20
tpr 99%  5.0-10.0  GeV: thr=0.18 fpr=0.36
tpr 99% 10.0-20.0  GeV: thr=0.23 fpr=0.38
tpr 99% 20.0-30.0  GeV: thr=0.23 fpr=0.47
tpr 99% 30.0-50.0  GeV: thr=0.32 fpr=0.56
tpr 99% 50.0-999.0 GeV: thr=0.43 fpr=0.67


tpr 90%  0.0-5.0   GeV: thr=0.77 fpr=0.07
tpr 90%  5.0-10.0  GeV: thr=0.82 fpr=0.12
tpr 90% 10.0-20.0  GeV: thr=0.81 fpr=0.15
tpr 90% 20.0-30.0  GeV: thr=0.77 fpr=0.22
tpr 90% 30.0-50.0  GeV: thr=0.74 fpr=0.37
tpr 90% 50.0-999.0 GeV: thr=0.73 fpr=0.61
"""
pt_edges=[0,5,10,20,30,50,999]
thr97=[0.54,0.52,0.52,0.52,0.51,0.50]
thr99=[0.23,0.18,0.23,0.23,0.32,0.43]
thr90=[0.77,0.82,0.81,0.77,0.74,0.73]
eff=[0.9,0.97,0.99]
rebin=hist.rebin(2)
h2pt=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=signal["GenEle/pt;1"].to_hist()[rebin]


pteff = TEfficiency(name="pt_eff", linewidth=3, xlabel="Gen $p_{T}$ [GeV]",xlim=(-5,105))

pteff.add(h2pt.integrate(2).integrate(0)[rebin],genpt,label="BDT>0")
for iteration,thr in enumerate([thr90,thr97,thr99]):
    hWP=0
    for idx,(low,high) in enumerate(zip(pt_edges[:-1],pt_edges[1:])):
        hWP=hWP+h2pt.integrate(2,hist.loc(low),hist.loc(high)).integrate(0,hist.loc(thr[idx]),2j)[rebin]
        plt.axvline(low,linestyle="--",color="red",linewidth=1)
        if iteration==0:
            plt.text(low,-0.15,f"{thr97[idx]:.2f}",fontsize=11,color="red")



    pteff.add(hWP,genpt,label=f"{eff[iteration]*100:.0f}% sig eff",linewidth=4)
    #pteff.add((h2pt.integrate(2).integrate(0)*eff[iteration])[rebin],genpt,label=f"{eff[iteration]:.2f}*BDT>0",color=plt.gca().lines[-1].get_color(),linestyle="--",alpha=0.6)

plt.text(-15,-0.15,"score cut (97%):",fontsize=10,color="red")
plt.text(low+8,-0.15,f"on online pt bins",fontsize=11,color="red")

pteff.add(signal["TkEleGenMatch/GenEle/pt;1"].to_hist()[rebin],genpt,label="TkEle",linewidth=4)


if save:
    pteff.save(folder+"pteff_097ptbins.pdf")

#%%


#!!RATE constant in binpt

rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,60),ylim=(8,50000),markersize=10,linewidth=4)


h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()

score_cuts=h2rate.axes[1].edges[:-1]

plt.text(-6.5,4.8,"score cut (97%):",fontsize=8,color="red")

edges=h2rate.axes[0].edges
rate.add(h2rate[:,0],label="BDT>0")
for iteration,thr in enumerate([thr90,thr97,thr99]):
    h=0
    for idx,(low,high) in enumerate(zip(pt_edges[:-1],pt_edges[1:])):

        bitmask=np.zeros(len(edges[:-1]))
        bitmask[np.bitwise_and(edges[:-1]>=low, edges[:-1]<high)]=1

        h=h+h2rate[:,hist.loc(thr[idx])]*bitmask
        if iteration==0:
            plt.text(low,4.8,f"{thr97[idx]:.2f}",fontsize=11,color="red")

    rate.add(h,label=f"{eff[iteration]*100:.0f}% sig eff")



rate.add(minbias["TkEle/rate_vs_ptcut;1"].to_hist(),label="TkEle")
for pt in pt_edges:
    plt.axvline(pt,linestyle="--",color="red",linewidth=1)

if save:
    rate.save(folder+"rate_097ptbins.pdf")

# %%
