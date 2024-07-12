
#%%
import importlib

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils import plots
from utils.utils import predict
from xgboost import XGBClassifier

train_test_split=sklearn.model_selection.train_test_split
importlib.reload(plots)

import mplhep as hep

hep.styles.cms.CMS["figure.autolayout"]=True
hep.style.use(hep.style.CMS)


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
]

df=pd.read_parquet(filename)

df_feat=df[features]
y=df["label"]
w=df["weight"]

X_train, X_test, y_train, y_test, w_train, w_test  = train_test_split(df_feat, y, w, test_size=0.2, random_state=666)



clf = XGBClassifier(n_estimators=11, max_depth=10,
                    eta=0.65,
                    reg_alpha=400,
                    reg_lambda=400,
                    objective="multi:softprob",
                    num_class=3,
                    tree_method="exact")



clf.fit(X_train, y_train, sample_weight=w_train,verbose=True, eval_set=[(X_test, y_test)], eval_metric="mlogloss")

#%%
df["score"]=1-clf.predict_proba(df_feat)[:,0]
plots.plot_pt_roc(clf,df)

ax2,_=plots.plot_best_pt_roc(clf,df,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))