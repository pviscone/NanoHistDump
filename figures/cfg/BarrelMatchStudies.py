from python.plotters import TEfficiency

#!--- pt eff
pteff = TEfficiency(name="pt_eff", xlabel="Gen $p_{T}$ [GeV]")
pteff.lazy_add(["CryCluGenMatch/GenEle/pt", "GenEle/pt"],label="CryClu-Gen")
pteff.lazy_add(["TkGenMatch/GenEle/pt", "GenEle/pt"], label="Tk-Gen")
pteff.lazy_add(["TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle/pt", "GenEle/pt"], label="Tk-Gen+CryClu-Gen")
pteff.lazy_add(["TkEleGenMatch/GenEle/pt", "GenEle/pt"], label="TkEle-Gen")
pteff.lazy_add(["TkCryCluGenMatch/CryCluGenMatchAll/GenEle/pt", "GenEle/pt"], label="Tk-CryClu-Gen")


# %%
#!--- eta eff
etaeff = TEfficiency(name="eta_eff", xlabel="Gen $\eta$",xlim=(-1.7,1.7))
etaeff.lazy_add(["CryCluGenMatch/GenEle/eta", "GenEle/eta"], label="CryClu-Gen")
etaeff.lazy_add(["TkGenMatch/GenEle/eta", "GenEle/eta"], label="Tk-Gen")
etaeff.lazy_add(["TkGenCryCluGenMatch/CryCluGenMatchAll/GenEle/eta", "GenEle/eta"], label="Tk-Gen+CryClu-Gen")
etaeff.lazy_add(["TkEleGenMatch/GenEle/eta", "GenEle/eta"], label="TkEle-Gen")
etaeff.lazy_add(["TkCryCluGenMatch/CryCluGenMatchAll/GenEle/eta", "GenEle/eta"], label="Tk-CryClu-Gen")



plots=[pteff,etaeff]
