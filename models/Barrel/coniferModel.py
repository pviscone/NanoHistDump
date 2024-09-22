# %%
import os
import sys
import numpy as np

sys.path.append("/afs/cern.ch/work/p/pviscone/conifer/conifer")
sys.path.append("../")

import conifer
import conifer.converters.xgboost
import matplotlib.pyplot as plt
import mplhep as hep
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.utils.extmath import softmax
from utils.bitscaler import BitScaler
from utils.utils import load_parquet

sys.path.append("../utils")
import plots

hep.style.use("CMS")

classes = 2

test_file = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/pandas_dataset/131Xv3_test.parquet"
model = f"barrel{classes}classes_131Xv3.json"

features = [
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

#scaler = BitScaler()
#scaler.load("scaler.npy")
scaler=None

df_test, dtest, _ = load_parquet(test_file, features, scaler=scaler, ptkey="CC_pt", label2=2 if classes > 2 else 1)

xgbmodel = xgb.Booster()
xgbmodel.load_model(model)

build = False

#%%
#!----------------------VIVADO ENVS----------------------!#
os.environ["PATH"] = "/home/Xilinx/Vivado/2023.1/bin:/home/Xilinx/Vitis_HLS/2023.1/bin:" + os.environ["PATH"]
os.environ["XILINX_AP_INCLUDE"] = "/opt/Xilinx/Vitis_HLS/2023.1/include"
os.environ["JSON_ROOT"] = "/afs/cern.ch/work/p/pviscone/conifer"

#!----------------------CFG----------------------!#
backend = "vivado"
score_precision = "ap_fixed<12,4,AP_RND_CONV,AP_SAT>"
input_precision = "ap_fixed<24,9,AP_RND_CONV,AP_SAT>"
threshold_precision = "ap_fixed<24,9,AP_RND_CONV,AP_SAT>"
if backend == "vivado":
    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["XilinxPart"] = "xcvu13p-flga2577-2-e"
    cfg["input_precision"] = input_precision
    cfg["threshold_precision"] = threshold_precision
    cfg["score_precision"] = score_precision
    cfg["ClockPeriod"] = 5.56


elif backend == "py":
    cfg = {"backend": "py", "output_dir": "dummy", "project_name": "dummy", "Precision": "float"}

elif backend == "cpp":
    cfg = conifer.backends.cpp.auto_config()
    cfg["input_precision"] = input_precision
    cfg["threshold_precision"] = threshold_precision
    cfg["score_precision"] = score_precision

cfg["OutputDir"] = f"conifer_{model.replace('.json', '')}"


def convert_and_evaluate(model, dmatrix, cfg, name, save=False):
    y = dmatrix.get_label()
    y[y == 2] = 1
    hls_model = conifer.converters.convert_from_xgboost(model, cfg)
    hls_model.compile()
    xgbpreds = 1 - model.predict(dmatrix)[:, 0] if classes > 2 else model.predict(dmatrix)
    hls_preds = hls_model.decision_function(dmatrix.get_data().toarray())
    hls_preds = 1 - softmax(hls_preds)[:, 0] if classes > 2 else 1/(1+np.exp(-hls_preds))
    fpr, tpr, _ = roc_curve(y, xgbpreds)
    hlsfpr, hlstpr, _ = roc_curve(y, hls_preds)
    plt.plot(fpr, tpr, label=f"XGBoost {name}")
    plt.plot(hlsfpr, hlstpr, label=f"{cfg['Precision']} {name}")
    plt.grid(True)
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if save:
        plt.savefig(save)
    return hls_model, xgbpreds, hls_preds


# %%
#!----------------------Conifer model----------------------!#
import os
os.makedirs(cfg["OutputDir"]+"/fig",exist_ok=True)

hlsmodel, df_test["XGBScore"], df_test["HLSScore"] = convert_and_evaluate(
    xgbmodel, dtest, cfg, "", save=cfg['OutputDir']+"/fig/conifer_rocs.pdf"
)

# %%
#!----------------------BUILD----------------------!#
if build:
    hlsmodel.build()

# cp report in fig

# %%

plots.plot_best_pt_roc(
    df_test,
    ptkey="CC_pt",
    score="HLSScore",
    thrs_to_select=[0.85, 0.7, 0.35, 0.3, 0.55, 0.85],
    save=cfg['OutputDir']+"/fig/hls_pt_roc.pdf",
)


plots.plot_best_pt_roc(
    df_test,
    ptkey="CC_pt",
    score="XGBScore",
    thrs_to_select=[0.85, 0.7, 0.35, 0.3, 0.55, 0.85],
    save=cfg['OutputDir']+"/fig/xgb_pt_roc.pdf",
)
