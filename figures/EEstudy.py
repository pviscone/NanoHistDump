# %%
import sys

sys.path.append("..")

import hist
import uproot

from python.plotters import TH1, TEfficiency

pu0 = uproot.open("../out/EEstudy_DoubleElectronsPU0_131Xv3Tc.root")

genmatch = TEfficiency(name="", xlabel="Gen $p_{T}$ [GeV]", lumitext="Endcap PU0", xlim=(-5, 100), rebin=2)
genmatch.add(pu0["HGCalCluGenMatchSelected/GenEle/pt"], pu0["GenEle/pt"], label="CluGenMatch")
pt_score = (
    pu0["HGCalCluGenMatch/pt_vs_puScore_vs_emScore"]
    .to_hist()
    .integrate(2, hist.loc(0.115991354), None)
    .integrate(1, None, hist.loc(0.4878136))
)
genmatch.add(pt_score, pu0["GenEle/pt"], label="LooseID WP")

pt_score_tk = (
    pu0["HGCalCluTkGenMatch/pt_vs_puScore_vs_emScore"]
    .to_hist()
    .integrate(2, hist.loc(0.115991354), None)
    .integrate(1, None, hist.loc(0.4878136))
)
genmatch.add(pu0["HGCalCluTkGenMatchSelected/HGCalCluGenMatch/GenEle/pt"], pu0["GenEle/pt"], label="TkCluGenMatch")
genmatch.add(pt_score_tk, pu0["GenEle/pt"], label="LooseID WP Tk Match")
# %%

n = TH1(name="n", xlabel="n", ylabel="Density", lumitext="Endcap PU0", rebin=1, log="y", density=True)
n.add(pu0["n/HGCalCluGenMatch"], label="Cluster per Gen")
n.add(pu0["n/HGCalCluTkGenMatch"], label="Tk per Cluster")
# %%
    