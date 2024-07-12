# %%
import copy
import os

import conifer
import conifer.converters.xgboost
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.utils.extmath import softmax
from utils import plots

hep.style.use("CMS")



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
    "abs_dPhi"
    ]

#conifer.model.load_model("full.conifer")

model_name="trytraining.json"


build=False


xgbmodel = xgb.XGBClassifier()
booster = xgb.Booster()
booster.load_model(model_name)
xgbmodel._Booster = booster
xgbmodel.load_model(model_name)



data=pd.read_parquet("131Xv3.parquet").sample(frac=0.3)


y=data["label"].to_numpy()
y[y==2]=1


# %%

#!----------------------VIVADO ENVS----------------------!#
os.environ["PATH"] = "/home/Xilinx/Vivado/2023.1/bin:/home/Xilinx/Vitis_HLS/2023.1/bin:" + os.environ["PATH"]
os.environ["XILINX_AP_INCLUDE"] = "/opt/Xilinx/Vitis_HLS/2023.1/include"
os.environ["JSON_ROOT"]="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/utils"

#!----------------------CFG----------------------!#
backend="cpp"

if backend=="vivado":
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
    cfg["Precision"] = "ap_fixed<64,32>"
    light_cfg = copy.deepcopy(cfg)
    cfg["OutputDir"] = "Fullprj_try"
    light_cfg["OutputDir"] = "Lightprj_try"
elif backend=="py":
    cfg={"backend" : "py", "output_dir" : "dummy", "project_name" : "dummy","Precision":"float"}
    light_cfg = cfg
elif backend=="cpp":
    cfg=conifer.backends.cpp.auto_config()
    cfg["Precision"] = "float"
    light_cfg = copy.deepcopy(cfg)
    cfg["OutputDir"] = "Fullprj_try"
    light_cfg["OutputDir"] = "Lightprj_try"


#%%
def convert_and_evaluate(model,data,cfg,name, save=False):
    hls_model = conifer.converters.convert_from_xgboost(model, cfg)
    hls_model.compile()
    xgbpreds = 1-model.predict_proba(data)[:,0]
    hls_preds = hls_model.decision_function(data.to_numpy())
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

model,data["FullXGBScore"],data["FullHLSScore"]=convert_and_evaluate(xgbmodel,data[features],cfg,"Full")

#%%
#!----------------------BUILD----------------------!#
if build:
    model.build()


# %%
#!----------------------VIVADO REPORTS----------------------!#
import hls4ml

hls4ml.report.vivado_report._find_reports("/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/Fullprj/my_prj/solution1","my_prj")


hls4ml.report.vivado_report._find_reports("/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/Lightprj/my_prj/solution1","my_prj")
# %%

data["score"]=data["FullHLSScore"]
plots.plot_best_pt_roc(model,data,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],save="../fig/hls_full_roc.pdf")

data["score"]=data["LightHLSScore"]
plots.plot_best_pt_roc(model,data,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],save="../fig/hls_light_roc.pdf")
