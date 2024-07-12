#%%
import importlib
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import xgboost as xgb
from utils import plots, utils
from utils.bitscaler import BitScaler

importlib.reload(plots)

import mplhep as hep
import numpy as np

hep.styles.cms.CMS["figure.autolayout"]=True
hep.style.use(hep.style.CMS)

save_model="light3_131Xv3.json"
save_model=False
load=False

train_file="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flatBDT/dataset/131Xv3_train.parquet"
test_file="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flatBDT/dataset/131Xv3_test.parquet"

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
    #Comment for light model
    #"dEta",
    #"dPhi",
    #"CryClu_isSS",
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

#!USA MINMAX SCALER INVECE (e vedi se si puó salvare)
range_map={
    "CryClu_pt":(0,120),
    "CryClu_ss":(0,1),
    "CryClu_relIso":(0,170),
    "CryClu_standaloneWP":(0,1),
    "CryClu_looseL1TkMatchWP":(0,1),
    "Tk_chi2RPhi":(0,200),
    "Tk_PtFrac":(0,1),
    "PtRatio":(0,51),
    "nMatch":(0,14),
    "abs_dEta":(0,0.03),
    "abs_dPhi":(0,0.3),
}

scaler=BitScaler()
scaler.fit(range_map)

#!Prova PCA e aggiungi scaler a load parquet
df_train,dtrain=utils.load_parquet(train_file,features,scaler=scaler,ptkey="CC_pt")
df_test,dtest=utils.load_parquet(test_file,features,scaler=scaler,ptkey="CC_pt")



#%%
def train(dtrain, dtest,save=False):
    params = {
        "tree_method": "hist",
        "max_depth": 10,
        "learning_rate": 0.65,
        "lambda": 1000,
        "alpha": 1000,
        #"colsample_bytree":0.9,
        "subsample":0.9,
        #"gamma":5,
        #"min_split_loss":5,
        "min_child_weight":80,
        #"enable_categorical": True,
        #"objective": "binary:logistic",
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    num_round = 9
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
    plots.plot_loss(eval_result, loss="mlogloss",save="fig/loss.pdf")

df_train["score"]=1-model.predict(dtrain)[:,0]
df_test["score"]=1-model.predict(dtest)[:,0]
#%%

rank=plots.plot_importance(model,save="fig")


#%%

plots.plot_scores(df_train["score"],df_train["label"],df_test["score"],df_test["label"],
                  func=lambda x: np.arctanh(x),
                  log=True,
                  bins=np.linspace(0,3.5,30),
                  save="fig/scores.pdf")
plots.plot_pt_roc(df_train,ptkey="CC_pt",save="fig/train_pt_roc.pdf")
plots.plot_pt_roc(df_test,ptkey="CC_pt",save="fig/test_pt_roc.pdf")

#%%


ax1,_=plots.plot_best_pt_roc(df_train,ptkey="CC_pt",thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
plt.savefig("fig/train_chosen_pt_roc.pdf")

ax2,_=plots.plot_best_pt_roc(df_test,ptkey="CC_pt",thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
plt.savefig("fig/test_chosen_pt_roc.pdf")
