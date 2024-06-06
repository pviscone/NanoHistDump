# %%
#!------------------------ IMPORTS ------------------------!#
import sys

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append("../../..")

from cfg.functions.matching import elliptic_match, match_to_gen
from cfg.functions.utils import set_name
from python.sample import Sample

hep.style.use("CMS")

#!------------------------ SETTINGS ------------------------!#
ellipse = [[0.03, 0.3]]
BarrelEta = 1.479
save = False

signal_path = "/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU200"
pu_path = "/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/MinBias"

schema = {
    "CaloEGammaCrystalClustersGCT": "CryClu",
    "DecTkBarrel": "Tk",
    "GenEl": "GenEle",
}

features=[
    "CryClu_pt",
    "CryClu_ss",#
    "CryClu_relIso",#
    "CryClu_isIso",
    "CryClu_isSS",
    "CryClu_isLooseTkIso",
    "CryClu_isLooseTkSS",
    "CryClu_brems",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_hitPattern",
    "Tk_pt",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RZ",
    "Tk_chi2RPhi",
    "dEta",
    "dPhi",
    "PtRatio",
    #Auxiliary (to remove before training)
    "GenEle_pt",
    "Tk_isReal",
    "CryClu_phi",
    "CryClu_eta",
    "evId"
]

#%%
#!--------------------- Collection building ---------------------!#
pu = Sample("", path=pu_path, tree_name="Events", scheme_dict=schema, nevents=91000).events
signal = Sample("", path=signal_path, tree_name="Events", scheme_dict=schema).events

def new_var(events):
    events["CryClu","ss"] = events.CryClu.e2x5/events.CryClu.e5x5
    events["CryClu","relIso"] = events.CryClu.isolation/events.CryClu.pt
    return events

signal = new_var(signal)
pu = new_var(pu)

signal["GenEle"] = signal.GenEle[np.abs(signal.GenEle.eta) < BarrelEta]
signal = signal[ak.num(signal.GenEle) > 0]


signal["CryCluGenMatch"] = match_to_gen(
    signal.GenEle, signal.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")), nested=True
)

mindpt_mask = ak.argmin(np.abs(signal["CryCluGenMatch"].dPt), axis=2, keepdims=True)

signal["CryCluGenMatch"] = ak.flatten(signal["CryCluGenMatch"][mindpt_mask], axis=2)

set_name(signal.CryCluGenMatch, "CryCluGenMatch")

signal["CluTk"] = elliptic_match(
    signal.CryCluGenMatch,
    signal.Tk,
    etaphi_vars=[["CryClu/eta", "CryClu/phi"], ["caloEta", "caloPhi"]],
    ellipse=ellipse,
)

pu = pu[ak.num(pu.GenEle) == 0]
pu["CluTk"] = elliptic_match(pu.CryClu, pu.Tk, etaphi_vars=[["eta", "phi"], ["caloEta", "caloPhi"]], ellipse=ellipse)

signal["CluTk","evId"]=np.arange(len(signal))
pu["CluTk","evId"]=np.arange(len(signal),len(pu)+len(signal))



# %%


def SF(pu_pt, sig_pt):
    def plot(*, pu_pt, sig_pt, ratio, smooth_ratio, interpol_func):
        fig, ax = plt.subplots(2, 1)
        ax[0].hist(pu_pt, bins=bins, density=True, histtype="step", label="PU", linewidth=2)
        ax[0].hist(sig_pt, bins=bins, density=True, histtype="step", label="signal", linewidth=2)
        ax[0].legend()
        ax[0].grid()
        ax[0].set_yscale("log")

        ax[1].stairs(ratio, edges=bins, color="black", label="Signal/PU", linewidth=2)
        ax[1].plot(centers, smooth_ratio, ".", color="red", label="smoothed", markersize=10)
        x = np.linspace(0, 120, 500)
        ax[1].plot(x, interpol_func(x), color="dodgerblue", label="interpolated", linewidth=2)
        ax[1].set_yscale("log")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_xlabel("CryClu Pt [GeV]")
        fig.savefig("../fig/ptSF.pdf")

    def closure(func,pu_pt,sig_pt):
        weights=func(pu_pt)
        fig,ax=plt.subplots()
        bins = np.array([22.75,38.75,65.75,111.5])
        bins=np.concatenate([np.linspace(1,15,20),bins])
        bins=np.linspace(1,120,150)
        sum_sig=len(sig_pt[sig_pt<40])
        sum_pu=np.sum(weights[pu_pt<40])
        ax.hist(sig_pt, density=False, histtype="step", label="signal", linewidth=2,bins=bins,weights=np.ones(len(sig_pt))/sum_sig)
        ax.hist(pu_pt, weights=weights/sum_pu, density=False, histtype="step", label="PU", linewidth=2,bins=bins)
        ax.grid()
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("CryClu Pt [GeV]")
        
        fig.savefig("../fig/ptClosure.pdf")

    pu_pt = np.array(ak.flatten(ak.flatten(ak.drop_none(pu_pt), axis=2)))
    sig_pt = np.array(ak.flatten(ak.flatten(ak.drop_none(sig_pt), axis=2)))
    #bins = bayesian_blocks(pu_pt, p0=1e-3)
    bins = np.array([22.75,38.75,65.75,111.5])
    bins=np.concatenate([np.linspace(1,16,30),bins])
    centers = (bins[1:] + bins[:-1]) / 2
    pu_hist = np.histogram(pu_pt, bins=bins,weights=np.ones_like(pu_pt)/len(pu_pt))[0]
    sig_hist = np.histogram(sig_pt, bins=bins,weights=np.ones_like(sig_pt)/len(sig_pt))[0]

    ratio = np.nan_to_num(sig_hist / pu_hist, 0)
    smooth_ratio = np.exp(lowess(exog=centers, endog=np.log(1e-18 + ratio), frac=0.3, it=100)[:, 1])
    interp_func = PchipInterpolator(centers, np.log(smooth_ratio))

    def interp(x):
        res = np.zeros_like(x)
        res[x > 50] = np.exp(interp_func(50))
        res[x < 50] = np.exp(interp_func(x[x < 50]))
        return res

    plot(pu_pt=pu_pt, sig_pt=sig_pt, ratio=ratio, smooth_ratio=smooth_ratio, interpol_func=interp)
    closure(interp,pu_pt,sig_pt)
    return interp


interp = SF(pu.CluTk.CryClu.pt, signal.CluTk.CryCluGenMatch.CryClu.pt)

# %%
sig_features=[]
bkg_features=[]
for feature in features:
    signal_feature=feature
    if "CryClu_" in feature or "GenEle_" in feature:
        signal_feature="CryCluGenMatch_"+feature
    sig_features.append(signal_feature)

    if "GenEle" in feature or "isReal" in feature:
        continue
    bkg_features.append(feature)


sig_dict={feature.split("CryCluGenMatch_")[-1]:np.array(ak.flatten(ak.flatten(ak.drop_none(signal.CluTk[*feature.split("_")]), axis=2))) for feature in sig_features}

bkg_dict={feature:np.array(ak.flatten(ak.flatten(ak.drop_none(pu.CluTk[*feature.split("_")]), axis=2))) for feature in bkg_features}


sig_df=pd.DataFrame(sig_dict)
bkg_df=pd.DataFrame(bkg_dict)

sig_df.loc[sig_df["CryClu_isLooseTkSS"]==2,"CryClu_isLooseTkSS"]=1
bkg_df.loc[bkg_df["CryClu_isLooseTkSS"]==2,"CryClu_isLooseTkSS"]=1

bkg_df["label"]=0
bkg_df["GenEle_pt"]=0

sig_df["label"]=sig_df["Tk_isReal"]
sig_df.loc[sig_df["label"]==0,"label"]=2
sig_df=sig_df.drop(columns=["Tk_isReal"])


sig_df["weight"]=1
bkg_df["weight"]=interp(bkg_df["CryClu_pt"])

sig_sum=sig_df[sig_df["CryClu_pt"]<40]["weight"].sum()
bkg_sum=bkg_df[bkg_df["CryClu_pt"]<40]["weight"].sum()

sig_df["weight"]=1/sig_sum
bkg_df["weight"]=bkg_df["weight"]/bkg_sum

df=pd.concat([sig_df,bkg_df])
df.to_parquet("../131Xv3.parquet")


corr=df.select_dtypes("number").corr()
corr_plot=sns.heatmap(corr, cmap="vlag",  linewidths=1,center=0, xticklabels=True, yticklabels=True)
plt.savefig("../fig/corr.pdf")
