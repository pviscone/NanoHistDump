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

filename = "../out/BarrelMatchStudies_DoubleElectrons_131Xv3.root"
file = uproot.open(filename)


# %%
n=file["TkCryCluGenMatch/Tk/isReal"].to_numpy()
nn=n[0][n[0]!=0]
import matplotlib.pyplot as plt
plt.pie(nn,labels=["Other","Electron", "PileUp"], autopct='%.2f%%')
