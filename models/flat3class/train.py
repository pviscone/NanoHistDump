
#%%
import importlib

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import xgboost as xgb
from utils import plots
from utils.utils import load_from_df, predict

train_test_split=sklearn.model_selection.train_test_split
importlib.reload(plots)

import mplhep as hep

hep.styles.cms.CMS["figure.autolayout"]=True
hep.style.use(hep.style.CMS)

"""
categorical=[
    "CryClu_isIso",
    "CryClu_isSS",
    "CryClu_isLooseTkIso",
    "CryClu_isLooseTkSS",
    "CryClu_brems",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_hitPattern",
]
"""

filename="131Xv3.parquet"
save_model="flat3class_131Xv3.json"
save_model=False
load=False
#load="flat3class_131Xv3.json"
#!TODO
#! Focal loss
#? ranking
#? Reinsert label 2
#? Multiclass
#? Save plots
#? Move rocs and assessments to other script
#? Fix the weight directly in the dataset
# Prova mlp
# Fai Efficienze e rate
#? Save model
# Aggiungere min e max dpt con altre tracce
# Tk pt?

features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "PtRatio",
    "nMatch",
    "abs_dEta",
    "abs_dPhi",
    #"dEta",
    #"dPhi",
    #"CryClu_isSS",
    #Comment for light model
    #"CryClu_isIso",
    #"CryClu_isLooseTkIso",
    #"CryClu_isLooseTkSS",
    #"CryClu_brems",
    #"Tk_hitPattern",
    #"Tk_nStubs",
    #"Tk_chi2Bend",
    #"Tk_chi2RZ",
    #
    #"Tk_pt",
    #"maxPtRatio_other",
    #"minPtRatio_other",
    #"meanPtRatio_other",
    #"stdPtRatio_other",
]

df=pd.read_parquet(filename)


#import numpy as np
#mask=np.bitwise_and(df["CryClu_pt"]<5,df["label"]==0)
#df.loc[mask,"weight"]=df.loc[mask,"weight"]/10
#df=df[df["CryClu_pt"]>5]

data,dtrain,dtest=load_from_df(df,features,label2=2,test_size=0.3)


def train(dtrain, dtest,save=False):
    params = {
        "tree_method": "exact",
        "max_depth": 10,
        "learning_rate": 0.65,
        "lambda": 400,
        "alpha": 400,
        #"enable_categorical": True,
        #"objective": "binary:logistic",
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    num_round = 11
    evallist = [(dtrain, "train"), (dtest, "eval")]
    eval_result = {}
    model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)
    if save:
        model.save_model(save)
    return model, eval_result


if load:
    model=xgb.Booster()
    model.load_model(load)
else:
    model,eval_result=train(dtrain,dtest,save=save_model)
    plots.plot_loss(eval_result,save="fig/loss.pdf")
data["score"]=1-predict(model,data,features)[:,0]
#%%
xgb.plot_importance(model,importance_type="gain",show_values=False)
plt.savefig("fig/importance_average_gain.pdf")
xgb.plot_importance(model,importance_type="weight",show_values=False)
plt.savefig("fig/importance_weight.pdf")

#%%
rank={}
for key in model.get_score():
    rank[key]=model.get_score(importance_type="weight")[key]*model.get_score(importance_type="gain")[key]
sorted_rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1])}

plt.barh(list(sorted_rank.keys()),width=sorted_rank.values())
plt.xlabel("Gain")
plt.savefig("fig/importance_gain.pdf")


#%%

plots.plot_scores(model,dtrain,dtest,log=False,save="fig/scores.pdf")
plots.plot_pt_roc(model,data,save="fig/pt_roc.pdf")

#%%
#plots.plot_best_pt_roc(model,data,eff=0.97,save="fig/best_pt_roc97.pdf")
#best_df=plots.plot_best_pt_roc(model,data,eff=[0.35,0.6,0.85,0.96,0.97,0.99],save="fig/best_pt_roc.pdf")
#best_df=plots.plot_best_pt_roc(model,data,thrs_to_select=[0.99,0.96,0.68,0.67,0.77,0.92],save="fig/pt_roc_tkEleEff.pdf")

#[0.87,0.65,0.25,0.15,0.1,0.072]
#plots.plot_best_pt_roc(model,data,thrs_to_select=[0.87,0.65,0.25,0.15,0.1,0.072],save="fig/pt_roc_tkEleRate.pdf")

#[0.85,0.7,0.35,0.3,0.55,0.85]
train,test=train_test_split(data,test_size=0.2,random_state=666)
ax1,_=plots.plot_best_pt_roc(model,test,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax1.text(0.3,0.5,"Test",fontsize=28)
plt.savefig("fig/chosen_pt_roc_test.pdf")

ax2,_=plots.plot_best_pt_roc(model,train,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax2.text(0.3,0.5,"Train",fontsize=28)
plt.savefig("fig/chosen_pt_roc_train.pdf")

# %%
ax2,_=plots.plot_best_pt_roc(model,data,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax2.text(0.3,0.5,"All",fontsize=28)
plt.savefig("fig/chosen_pt_roc_all.pdf")