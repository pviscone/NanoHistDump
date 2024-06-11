#%%
import importlib

import matplotlib.pyplot as plt
import xgboost as xgb
from utils import plots
from utils.utils import load_data, predict

importlib.reload(plots)

import mplhep as hep

hep.styles.cms.CMS["figure.autolayout"]=True
hep.style.use(hep.style.CMS)

features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_isIso",
    "CryClu_isSS",
    "CryClu_isLooseTkIso",
    "CryClu_isLooseTkSS",
    "CryClu_brems",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_hitPattern",
    #"Tk_pt",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RZ",
    "Tk_chi2RPhi",
    "dEta",
    "dPhi",
    "PtRatio",
    "nMatch"
]

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
load="flat3class_131Xv3.json"
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



def train(dtrain, dtest,save=False):
    params = {
        "tree_method": "exact",
        "max_depth": 10,
        "learning_rate": 0.4,
        "lambda": 400,
        "alpha": 400,
        #"enable_categorical": True,
        #"objective": "binary:logistic",
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    num_round = 25
    evallist = [(dtrain, "train"), (dtest, "eval")]
    eval_result = {}
    model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)
    if save:
        model.save_model(save)
    return model, eval_result


data,dtrain,dtest=load_data(filename,features,label2=2)

if load:
    model=xgb.Booster()
    model.load_model(load)
else:
    model,eval_result=train(dtrain,dtest,save=save_model)
    plots.plot_loss(eval_result,save="fig/loss.pdf")
#%%
data["score"]=1-predict(model,data,features)[:,0]
xgb.plot_importance(model,importance_type="gain",values_format="{v:.0f}")
plt.savefig("fig/importance_gain.pdf")
xgb.plot_importance(model,importance_type="weight",values_format="{v:.0f}")
plt.savefig("fig/importance_weight.pdf")

plots.plot_scores(model,dtrain,dtest,log=False,save="fig/scores.pdf")
plots.plot_pt_roc(model,data,save="fig/pt_roc.pdf")
plots.plot_best_pt_roc(model,data,eff=0.97,save="fig/best_pt_roc97.pdf")
best_df=plots.plot_best_pt_roc(model,data,eff=[0.5,0.7,0.9,0.95,0.97,0.99],save="fig/best_pt_roc.pdf")

