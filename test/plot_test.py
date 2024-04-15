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
#!--- pt eff
pteff = TEfficiency(xlabel="Gen $p_{T}$ [GeV]")
pteff.add(file["CryCluGenMatch/GenEle/pt"], file["GenEle/pt"],label="CryClu-Gen")
pteff.add(file["TkGenMatch/GenEle/pt"], file["GenEle/pt"], label="Tk-Gen")
pteff.add(file["TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle/pt"], file["GenEle/pt"], label="Tk-Gen+CryClu-Gen")
pteff.add(file["TkEleGenMatch/GenEle/pt"], file["GenEle/pt"], label="TkEle-Gen")
pteff.add(file["TkCryCluGenMatch/CryCluGenMatchAll/GenEle/pt"], file["GenEle/pt"], label="Tk-CryClu-Gen")

pteff.save("pt_eff.pdf")
# %%
#!--- eta eff
etaeff = TEfficiency(xlabel="Gen $\eta$",xlim=(-1.7,1.7))
etaeff.add(file["CryCluGenMatch/GenEle/eta"], file["GenEle/eta"], label="CryClu-Gen")
etaeff.add(file["TkGenMatch/GenEle/eta"], file["GenEle/eta"], label="Tk-Gen")
etaeff.add(file["TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle/eta"], file["GenEle/eta"], label="Tk-Gen+CryClu-Gen")
etaeff.add(file["TkEleGenMatch/GenEle/eta"], file["GenEle/eta"], label="TkEle-Gen")
etaeff.add(file["TkCryCluGenMatch/CryCluGenMatchAll/GenEle/eta"], file["GenEle/eta"], label="Tk-CryClu-Gen")

etaeff.save("eta_eff.pdf")
