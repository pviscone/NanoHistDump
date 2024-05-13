#%%
import importlib
import sys

import uproot

sys.path.append("../..")
import python.plotters

importlib.reload(python.plotters)
TH1 = python.plotters.TH1
TEfficiency = python.plotters.TEfficiency
TH2 = python.plotters.TH2
TRate= python.plotters.TRate

folder="../fig/BDT_noWeight_131Xv3/"
filename_signal = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_noWeight_DoubleElectrons_131Xv3.root"
filename_minbias = "/data/pviscone/PhD-notes/submodules/NanoHistDump/out/BDT_noWeight_MinBias_131Xv3.root"
signal = uproot.open(filename_signal)
minbias=uproot.open(filename_minbias)
selected_cuts=[0,0.4,0.8]
#%%
#!------------------pt-------------------
h2pt=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=signal["GenEle/pt;1"].to_hist()

bdt_cuts=h2pt.axes[0].edges
pteff = TEfficiency(name="pt_eff", linewidth=3, xlabel="Gen $p_{T}$ [GeV]")
for n_bin,cut in enumerate(bdt_cuts[:-1]):
    if cut not in selected_cuts:
        continue
    integrated=h2pt.integrate(0,n_bin,2j)
    pteff.add(integrated,genpt,label=f"BDT>{cut:.2f}")
pteff.add(signal["TkEleGenMatch/GenEle/pt;1"].to_hist(),genpt,label="TkEle",linestyle="--",linewidth=4,color="black")
pteff.save(folder+"pt_eff.pdf")
#%%

#!-------------------eta-------------------!#
h2eta=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_eta;1"].to_hist()
geneta=signal["GenEle/eta;1"].to_hist()

etaeff = TEfficiency(name="eta_eff", linewidth=3, xlabel="Gen $\eta$")
for n_bin,cut in enumerate(bdt_cuts[:-1]):
    if cut not in selected_cuts:
        continue
    integrated=h2eta.integrate(0,n_bin,2j)
    etaeff.add(integrated,geneta,label=f"BDT>{cut:.2f}")
etaeff.add(signal["TkEleGenMatch/GenEle/eta;1"].to_hist(),geneta,label="TkEle",linestyle="--",color="black",linewidth=4)
etaeff.save(folder+"eta_eff.pdf")


#%%

rate = TRate(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,60), ylim=(8,50000),markersize=10,linewidth=4)
#!-------------------rate-------------------!#

h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
tkelerate=minbias["TkEleGenMatch/GenEle/pt;1"].to_hist()
score_cuts=h2rate.axes[1].edges[:-1]
for idx,cuts in enumerate(score_cuts):
    if cuts not in selected_cuts:
        continue
    rate.add(h2rate[:,idx],label=f"BDT score>{cuts:.2f}")

rate.add(minbias["TkEle/rate_vs_ptcut;1"].to_hist(),label="TkEle")

rate.save(folder+"rate.pdf")
# %%
