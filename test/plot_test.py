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
h = TH1()
h.add(file["CryCluGenMatch/GenEle/phi"], label="GenEle")
h.add(file["CryCluGenMatch/CryClu/phi"], label="CryClu")


# %%

e = TEfficiency()
e.add(file["TkCryCluGenMatchVertex/CryCluGenMatchAllVertex/GenEle/pt;1"], file["GenEle/pt"], label="Vertex")
e.add(file["TkCryCluGenMatch/CryCluGenMatchAll/GenEle/pt"], file["GenEle/pt"], label="Calo")
#%%

t2=TH2(log="z")
t2.add(file["n/TkCryCluGenMatchAllPt_vs_TkCryCluGenMatchAll"])

