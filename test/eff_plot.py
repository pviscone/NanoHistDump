# %%
import importlib
import sys

import mplhep as hep
import uproot
from rich.traceback import install

#install(show_locals=False)


sys.path.append("..")
import python.plotters

importlib.reload(python.plotters)
TH1 = python.plotters.TH1
TEfficiency = python.plotters.TEfficiency
TH2 = python.plotters.TH2


hep.style.use("CMS")

filename1 = "../out/BDT_DoubleElectrons_131Xv3.root"
filename2 = "../out/BDT_MinBias_131Xv3.root"
signal = uproot.open(filename1)

#%%
#!-------------------pt-------------------!#
h2pt=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_pt;1"].to_hist()
genpt=signal["GenEle/pt;1"].to_hist()

bdt_cuts=h2pt.axes[0].edges
pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]")
for n_bin,cut in enumerate(bdt_cuts[:-1]):
    integrated=h2pt.integrate(0,n_bin,2j)
    pteff.add(integrated,genpt,label=f"BDT>{cut:.2f}")
pteff.add(signal["TkEleGenMatch/GenEle/pt;1"].to_hist(),genpt,label="TkEle",linestyle="--")
pteff.save("pt_eff.pdf")
#%%

#!-------------------eta-------------------!#
h2eta=signal["TkCryCluGenMatch-GenEle/BDTscore_vs_eta;1"].to_hist()
geneta=signal["GenEle/eta;1"].to_hist()

etaeff = TEfficiency(name="eta_eff", xlabel="Gen $\eta$")
for n_bin,cut in enumerate(bdt_cuts[:-1]):
    integrated=h2eta.integrate(0,n_bin,2j)
    etaeff.add(integrated,geneta,label=f"BDT>{cut:.2f}")
etaeff.add(signal["TkEleGenMatch/GenEle/eta;1"].to_hist(),geneta,label="TkEle",linestyle="--")
etaeff.save("eta_eff.pdf")


#%%
rate = TH1(name="rate_vs_pt", xlabel="Online $p_T$ cut [GeV]", ylabel="Rate [kHz]", log="y",xlim=(0,50))
#!-------------------rate-------------------!#
minbias=uproot.open(filename2)
h2rate=minbias["TkCryCluMatch/rate_pt_vs_score;1"].to_hist()
score_cuts=h2rate.axes[1].edges[:-1]
for idx,cuts in enumerate(score_cuts):
    rate.add(h2rate[:,idx],label=f"BDT score>{cuts:.2f}")
rate.save("rate.pdf")