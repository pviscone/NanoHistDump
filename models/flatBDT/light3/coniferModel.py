# %%
import copy
import os
import sys
import importlib

sys.path.append("/afs/cern.ch/work/p/pviscone/conifer/conifer")
sys.path.append("../")

import conifer
import conifer.converters.xgboost
import matplotlib.pyplot as plt
import mplhep as hep
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.utils.extmath import softmax
from utils.utils import load_parquet
from utils.bitscaler import BitScaler


sys.path.append("../utils")
import plots




hep.style.use("CMS")

test_file="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flatBDT/dataset/131Xv3_test.parquet"
model="light3_131Xv3.json"

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

scaler=BitScaler()
scaler.load("scaler.npy")

df_test,dtest=load_parquet(test_file,features,scaler=scaler,ptkey="CC_pt")

xgbmodel = xgb.Booster()
xgbmodel.load_model(model)

build=False



#!----------------------VIVADO ENVS----------------------!#
os.environ["PATH"] = "/home/Xilinx/Vivado/2023.1/bin:/home/Xilinx/Vitis_HLS/2023.1/bin:" + os.environ["PATH"]
os.environ["XILINX_AP_INCLUDE"] = "/opt/Xilinx/Vitis_HLS/2023.1/include"
os.environ["JSON_ROOT"]="/afs/cern.ch/work/p/pviscone/conifer"

#!----------------------CFG----------------------!#
backend="cpp"

if backend=="vivado":
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
    cfg["Precision"] = "ap_fixed<64,32>"


elif backend=="py":
    cfg={"backend" : "py", "output_dir" : "dummy", "project_name" : "dummy","Precision":"float"}

elif backend=="cpp":
    cfg=conifer.backends.cpp.auto_config()
    cfg["Precision"] = "float"

cfg["OutputDir"] = "conifer_prj"



def convert_and_evaluate(model,dmatrix, cfg, name, save=False):
    y=dmatrix.get_label()
    y[y==2]=1
    hls_model = conifer.converters.convert_from_xgboost(model, cfg)
    hls_model.compile()
    xgbpreds = 1-model.predict(dmatrix)[:,0]
    hls_preds = hls_model.decision_function(dmatrix.get_data().toarray())
    hls_preds = 1-softmax(hls_preds)[:,0]
    fpr, tpr, _ = roc_curve(y, xgbpreds)
    hlsfpr, hlstpr, _ =roc_curve(y,hls_preds)
    plt.plot(fpr, tpr, label=f"XGBoost {name}")
    plt.plot(hlsfpr, hlstpr, label=f"{cfg['Precision']} {name}")
    plt.grid(True)
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if save:
        plt.savefig(save)
    return hls_model,xgbpreds,hls_preds



#%%
#!----------------------Light model----------------------!#

hlsmodel,df_test["XGBScore"],df_test["HLSScore"]=convert_and_evaluate(xgbmodel,dtest,cfg,"",save="fig/conifer/rocs.pdf")

#%%
#!----------------------BUILD----------------------!#
if build:
    hlsmodel.build()

# %%


plots.plot_best_pt_roc(df_test,ptkey="CC_pt",score="HLSScore",thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],save="fig/conifer/hls_roc.pdf")


plots.plot_best_pt_roc(df_test,ptkey="CC_pt",score="XGBScore",thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],save="fig/xgb_roc.pdf")



# %%
%load_ext autoreload
%autoreload 2

importlib.reload(plots)
import plots


inputs={key:df_test[key].to_numpy() for key in [*features,"XGBScore"]}
plots.profile_int_dec(inputs,sign_prop=True)
plt.savefig("fig/conifer/input_profile.pdf")
#!profila threshold e values
#! prova 2 interi 10 decimali
#!prova pca



# %%
import numpy as np



def model_profile(model,what="value"):
    feat_map=model._ensembleDict["feature_names"]
    res={}
    def update_res(tree):
        arr=np.array(getattr(tree,what))
        arr=arr[np.array(tree.feature)==feat_map[feat]]
        res[feat]=np.concatenate([res[feat],arr])

    for feat in feat_map:
        res[feat]=np.array([])
        for tree in model.trees:
            if model.n_classes>2:
                for subtree in tree:
                    update_res(subtree)
            else:
                update_res(tree)


    return res


val=model_profile(hlsmodel,"value")
thr=model_profile(hlsmodel,"threshold")


# %%
plots.profile_int_dec(val,sign_prop=False)
plt.savefig("fig/conifer/model_value_profile.pdf")
plots.profile_int_dec(thr,sign_prop=False)
plt.savefig("fig/conifer/model_threshold_profile.pdf")