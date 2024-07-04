#%%
import importlib
import os

import matplotlib.pyplot as plt
import xgboost as xgb
from utils import plots
from utils.utils import load_data

importlib.reload(plots)

import conifer
import mplhep as hep
from sklearn.metrics import roc_curve
from sklearn.utils.extmath import softmax
import ydf
from sklearn.model_selection import train_test_split
import pandas as pd

hep.styles.cms.CMS["figure.autolayout"]=True
hep.style.use(hep.style.CMS)

filename="131Xv3.parquet"


#!----------------------VIVADO ENVS----------------------!#
os.environ["PATH"] = "/home/Xilinx/Vivado/2023.1/bin:/home/Xilinx/Vitis_HLS/2023.1/bin:" + os.environ["PATH"]
os.environ["XILINX_AP_INCLUDE"] = "/opt/Xilinx/Vitis_HLS/2023.1/include"
os.environ["JSON_ROOT"]="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/utils"



features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    #"CryClu_isIso",
    "CryClu_isSS",
    #"CryClu_isLooseTkIso",
    #"CryClu_isLooseTkSS",
    #"CryClu_brems",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    #"Tk_hitPattern",
    #"Tk_nStubs",
    #"Tk_chi2Bend",
    #"Tk_chi2RZ",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "dEta",
    "dPhi",
    "PtRatio",
    "nMatch",
    #"Tk_pt",
    #"maxPtRatio_other",
    #"minPtRatio_other",
    #"meanPtRatio_other",
    #"stdPtRatio_other",
]

lib="ydf"
def train(depth,dtrain, dtest):
    if lib=="xgboost":
        params = {
            "tree_method": "exact",
            "max_depth": depth,
            "learning_rate": 0.65,
            "lambda": 400,
            "alpha": 400,
            #"enable_categorical": True,
            #"objective": "binary:logistic",
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
        }
        num_round = 12
        evallist = [(dtrain, "train"), (dtest, "eval")]
        eval_result = {}
        model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)
        return model, eval_result
    if lib=="ydf":
        model = ydf.GradientBoostedTreesLearner(label="label",
                                                task=ydf.Task.CLASSIFICATION,
                                                num_trees=12,
                                                max_depth=depth,
                                                shrinkage=0.65,
                                                loss="MULTINOMIAL_LOG_LIKELIHOOD",
                                                l1_regularization=400.,
                                                l2_regularization=400.,
                                                validation_ratio=0.2,
                                                ).train(dtrain)
        return model
    return None


def convert_and_evaluate(model,data,y,cfg):
    if lib=="xgboost":
        hls_model = conifer.converters.convert_from_xgboost(model, cfg)
        try:
            hls_model.compile()
        except:
            pass
        xgbpreds = 1-model.predict(data)[:,0]
        hls_preds = hls_model.decision_function(data.get_data().toarray())

    elif lib=="ydf":
        hls_model = conifer.converters.convert_from_ydf(model, cfg)
        try:
            hls_model.compile()
        except:
            pass
        xgbpreds = 1-model.predict(data)[:,0]
        hls_preds = hls_model.decision_function(data.drop("label",axis=1).to_numpy())

    hls_preds = 1-softmax(hls_preds)[:,0]
    fpr, tpr, _ = roc_curve(y, xgbpreds)
    hlsfpr, hlstpr, _ =roc_curve(y,hls_preds)
    plt.figure()
    plt.plot(fpr, tpr, label="XGBoost")
    plt.plot(hlsfpr, hlstpr, label=f"{cfg['Precision']}")
    plt.grid(True)
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"depth {depth}")

    return hls_model,xgbpreds,hls_preds

#%%
#!----------------------CFG----------------------!#
backend="py"

if backend=="vivado":
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
    cfg["Precision"] = "ap_fixed<64,32>"
    cfg["OutputDir"] = "Lightprj_try"
elif backend=="py":
    cfg={"backend" : "py", "output_dir" : "dummy", "project_name" : "dummy","Precision":"float"}
elif backend=="cpp":
    cfg=conifer.backends.cpp.auto_config()
    cfg["Precision"] = "float"
    cfg["OutputDir"] = "Lightprj_try"

if lib=="xgboost":
    _,dtrain,dtest=load_data(filename,features,label2=2,test_size=0.2)
    y=dtest.get_label()
    y[y==2]=1
elif lib=="ydf":
    features=[*features,"label"]
    dtrain,dtest=train_test_split(pd.read_parquet(filename)[features],test_size=0.2)
    y=dtest["label"].to_numpy()
    y[y==2]=1



for depth in [3,5,7,10]:
    if lib=="xgboost":
        xgbmodel,eval_result=train(depth,dtrain,dtest)
        plots.plot_loss(eval_result)
        model,_,_=convert_and_evaluate(xgbmodel,dtest,y,cfg)
    elif lib=="ydf":
        model=train(depth,dtrain,dtest)
        model,_,_=convert_and_evaluate(model,dtest,y,cfg)

# %%

def predict(model,data):
    return softmax(model.decision_function(data.get_data().toarray()))
