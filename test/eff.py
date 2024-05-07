# %%
import importlib
import sys

sys.path.append("..")

import xgboost as xgb

import python.sample
from cfg.BDT import define

BarrelEta = 1.479
model=xgb.Booster()
model.load_model("/data/pviscone/PhD-notes/submodules/NanoHistDump/models/xgboost/BDT_131Xv3.json")

importlib.reload(python.sample)
Sample = python.sample.Sample

signal_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/DoubleElectron"
pu_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/MinBias"


scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

cryclu_path="CryClu/"
tk_path="Tk/"
couple_path=""
features_minbias=[
    cryclu_path+"standaloneWP",
    cryclu_path+"showerShape",
    cryclu_path+"isolation",
    tk_path+"hitPattern",
    tk_path+"nStubs",
    tk_path+"chi2Bend",
    tk_path+"chi2RPhi",
    tk_path+"chi2RZ",
    couple_path+"dEtaCryClu",
    couple_path+"dPhiCryClu",
    couple_path+"dPtCryClu",
]
features=["CryCluGenMatch/"+feat if feat.startswith("CryClu/") else feat for feat in features_minbias]


signal=Sample("signal", path=signal_path, tree_name="Events", scheme_dict=scheme).events
pu=Sample("pu", path=pu_path, tree_name="Events", scheme_dict=scheme,nevents=40000).events

#%%
signal=define(signal,"DoubleElectrons")
pu=define(pu,"MinBias")

# %%
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import mplhep as hep
import hist
from sklearn.metrics import roc_curve

hep.style.use("CMS")

def flat(arr):
    return ak.flatten(ak.drop_none(arr))

#%%%#!ROC
signal_score=flat(signal["TkCryCluGenMatch","BDTscore"])
pu_score=flat(pu["TkCryCluMatch","BDTscore"])

y=np.concatenate([np.ones(len(signal_score)),np.zeros(len(pu_score))])
scores=np.concatenate([signal_score,pu_score])

fp, tp, thr = roc_curve(y,scores)
plt.plot(fp,tp)

idx_04=np.argmin(np.abs(thr-0.4))
plt.plot(fp[idx_04],tp[idx_04],"ro")
plt.grid()


#%%#!! Efficiency

signal_score=flat(signal["TkCryCluGenMatch","BDTscore"])


score_cut=0.4
gen_pt=flat(signal["GenEle","pt"])
match_pt=flat(signal["TkCryCluGenMatch","CryCluGenMatch","GenEle","pt"])
bins=np.linspace(0,105,105)
hgen=hist.Hist(hist.axis.Variable(bins))
hmatch=hist.Hist(hist.axis.Variable(bins))
hmatchcut=hist.Hist(hist.axis.Variable(bins))
hgen.fill(gen_pt)
hmatch.fill(match_pt)
hmatchcut.fill(match_pt[signal_score>score_cut])

(hmatch/hgen).plot(label="Match")

((hmatch/hgen)*0.9445).plot(label="Match Scaled *0.9445")
(hmatchcut/hgen).plot(label=f"Match BDT>{score_cut}")
plt.legend()
plt.grid()

#%%#!RATE
n_ev=len(pu)
freq_x_bx=2760.0*11246/1000
pt=ak.drop_none(pu["TkCryCluMatch","CryClu","pt"])
score=ak.drop_none(pu["TkCryCluMatch","BDTscore"])

hist_obj=hist.Hist(hist.axis.Regular(100,0,100),hist.axis.Variable(np.array([0,0.2,0.4,0.6,0.8,1.0])),)

score_cuts=hist_obj.axes[1].edges[:-1]
score_centers=hist_obj.axes[1].centers
for score_idx,score_cut in enumerate(score_cuts):
    pt_max=ak.drop_none(ak.max(pt[score>score_cut],axis=1))
    for thr,pt_bin_center in zip(hist_obj.axes[0].edges, hist_obj.axes[0].centers):
        hist_obj.fill(pt_bin_center,score_centers[score_idx], weight=ak.sum(pt_max>=thr)/n_ev)


# %%
#!XGBWRAPPER

cryclu_path="CryClu/"
tk_path="Tk/"
couple_path=""
features_minbias=[
    cryclu_path+"standaloneWP",
    cryclu_path+"showerShape",
    cryclu_path+"isolation",
    tk_path+"hitPattern",
    tk_path+"nStubs",
    tk_path+"chi2Bend",
    tk_path+"chi2RPhi",
    tk_path+"chi2RZ",
    couple_path+"dEtaCryClu",
    couple_path+"dPhiCryClu",
    couple_path+"dPtCryClu",
]
features=["CryCluGenMatch/"+feat if feat.startswith("CryClu/") else feat for feat in features_minbias]


import xgboost as xgb

from python.inference import xgb_wrapper


model=xgb.Booster()
model.load_model("/data/pviscone/PhD-notes/submodules/NanoHistDump/models/xgboost/BDT_131Xv3.json")

from cfg.functions.matching import match_obj_to_obj,match_to_gen,match_obj_to_couple
from cfg.functions.utils import set_name

pu["match"] = match_obj_to_obj(
    pu.CryClu, pu.Tk, etaphi_vars=(("eta", "phi"), ("caloEta", "caloPhi")),nested=True
)
pu["match", "dEtaCryClu"] = pu["match", "dEta"]
pu["match", "dPhiCryClu"] = pu["match", "dPhi"]
pu["match", "dPtCryClu"] = pu["match", "dPt"]
pu["match","BDTscore"]=xgb_wrapper(model, pu["match"],features_minbias,nested=True)



# %%

signal["match"] = match_obj_to_couple(
    signal.Tk, signal.CryCluGenMatch, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi")),nested=True
)

signal["match","BDTscore"]=xgb_wrapper(model, signal["match"],features,nested=True)
#%%
signal_score=flat(ak.flatten(signal["match","BDTscore"],axis=2))
pu_score=flat(ak.flatten(pu["match","BDTscore"],axis=2))

y=np.concatenate([np.ones(len(signal_score)),np.zeros(len(pu_score))])
scores=np.concatenate([signal_score,pu_score])

fp, tp, thr = roc_curve(y,scores)
plt.plot(fp,tp)

idx_04=np.argmin(np.abs(thr-0.4))
plt.plot(fp[idx_04],tp[idx_04],"ro")
plt.grid()




#%% #!WRAPPER TEST

signal["TkCryCluGenMatch"] = match_obj_to_couple(
            signal.Tk, signal.CryCluGenMatch, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi")),nested=True
        )

pu["TkCryCluMatch"] = match_obj_to_obj(
    pu.CryClu, pu.Tk, etaphi_vars=(("eta", "phi"), ("caloEta", "caloPhi")),nested=True
)
pu["TkCryCluMatch", "dPtCryClu"] = pu["TkCryCluMatch", "dPt"]
pu["TkCryCluMatch", "dEtaCryClu"] = pu["TkCryCluMatch", "dEta"]
pu["TkCryCluMatch", "dPhiCryClu"] = pu["TkCryCluMatch", "dPhi"]
#%%

nested=True
events=pu["TkCryCluMatch"]
layout_template=pu.TkCryCluMatch.dPtCryClu.layout
for idx, feature in enumerate(features_minbias):
    feature_list = feature.split("/")

    if feature_list[-1] != model.feature_names[idx].split("_")[-1]:
        raise ValueError(f"Feature name mismatch: {feature} instead of {model.feature_names[idx]}")

    array=events[*feature_list]
    array=ak.drop_none(array)
    if nested:
        array=ak.flatten(events[*feature_list])
    array=ak.drop_none(array)
    array = ak.flatten(array).to_numpy(allow_missing=False)[:, None]
    if idx == 0:
        matrix = array
    else:
        matrix = np.concatenate((matrix, array), axis=1)

dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_names)
scores = model.predict(dmatrix)


if nested:
    def recur(layout_template, arr):
        if "_content" in layout_template.__dir__():
            layout_template._content=recur(layout_template._content,arr)

        else:
            layout_template=arr
        return layout_template

    layout_template=recur(layout_template,ak.contents.NumpyArray(scores))
    awkpu_scores=ak.Array(layout_template)



# %%
