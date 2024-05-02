# %%
import importlib
import sys

import mplhep as hep
import uproot
from rich.traceback import install

install(show_locals=True)


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
pu = uproot.open(filename2)


# %%
#!Rate vs eff
N_pu = pu["triggers/maxBDT"].to_numpy()[0].sum()
N_signal = signal["triggers/maxBDT"].to_numpy()[0].sum()

score_list = [key.split("/")[-1].split(";")[0].split("isBDT")[-1] for key in pu.keys() if "isBDT" in key]

fpr=[]
tpr=[]
for score in score_list:
    fpr.append(pu[f"triggers/isBDT{score};1"].to_numpy()[0][-1]/N_pu)
    tpr.append(signal[f"triggers/isBDT{score};1"].to_numpy()[0][-1]/N_signal)

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.array(fpr)*2760.0*11246/1000, tpr)
plt.grid()
plt.xlabel("Rate [kHz]")
plt.ylabel("Electron Efficiency")


#%%
hep.histplot((pu["triggers/maxBDT"]).to_hist(),density=True,label="PU")
hep.histplot((signal["triggers/maxBDT"]).to_hist(),density=True,label="Signal")
plt.grid()
plt.ylabel("Density")
plt.legend()