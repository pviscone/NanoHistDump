# %%

import sys
sys.path.append("..")

import uproot

from python.plotters import TEfficiency,TH1

files={
    "M2":"DP_HAHM_M2_106X_setupSOS2023.root",
    "M5":"DP_HAHM_M5_106X_setupSOS2023.root",
    "M15":"DP_HAHM_M15_106X_setupSOS2023.root",

}

""" pteff = TEfficiency(
        name="pt_eff", xlabel="Gen $p_{T}$ [GeV]",  linewidth=3, rebin=4,
        cmstext="Simulation", lumitext="",yerr=False
    )

etaeff = TEfficiency(
        name="eta_eff", xlabel="Gen $\eta$",  linewidth=3, rebin=4,
        cmstext="Simulation", lumitext="",yerr=False
    ) """

pt0=TH1(name="Leading GenEle pT", xlabel="Leading Gen $p_{T}$ [GeV]", xlim=(-0.5,20),  linewidth=3, rebin=1,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
pt1=TH1(name="Subleading GenEle pT", xlabel="Subleading Gen $p_{T}$ [GeV]", xlim=(-0.5,20),  linewidth=3, rebin=1,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
ZdPt=TH1(name="DiEle pT", xlabel="DiEle $p_{T}$ [GeV]", xlim=(-0.5,30), linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
ZdMass=TH1(name="DiEle mass", xlabel="DiEle mass [GeV]", xlim=(-0.5,20),  linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
dr=TH1(name="GenEle $\Delta R$", xlabel="GenEle $\Delta R$",  linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
dphi=TH1(name="GenEle $\Delta \phi$", xlabel="GenEle $\Delta \phi$", linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
deta=TH1(name="GenEle $\Delta \eta$", xlabel="GenEle $\Delta \eta$",  linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)
dangle=TH1(name="GenEle $\Delta angle$", xlabel="GenEle $\Delta angle$", linewidth=3, rebin=2,cmstext="Simulation", lumitext="",density=True, alpha=0.85)


for mass, file in files.items():
    rfile=uproot.open(file)

    #pteff.add(rfile["LepGenMatch/GenEle/pt"], rfile["GenEle/pt;1"], label=mass)
    #etaeff.add(rfile["LepGenMatch/GenEle/eta"], rfile["GenEle/eta;1"], label=mass)
    pt0.add(rfile["GenEle/pt[0]"], label=mass)
    pt1.add(rfile["GenEle/pt[1]"], label=mass)

    ZdPt.add(rfile["Zd/pt"], label=mass)
    ZdMass.add(rfile["Zd/mass"], label=mass)
    dr.add(rfile["Zd/dr"], label=mass)
    dphi.add(rfile["Zd/dphi"], label=mass)
    deta.add(rfile["Zd/deta"], label=mass)
    dangle.add(rfile["Zd/dangle"], label=mass)





# %%
