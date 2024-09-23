
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

train_file = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/pandas_dataset/131Xv3_train.parquet"
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
#df_train, dtrain, _ = load_parquet(train_file, features, scaler=scaler, ptkey="CC_pt", label2=2 if classes > 2 else 1)
#df=pd.concat([df_train,df_test])

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
    cfg["Precision"] = "ap_fixed<24,9>"
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
    plt.plot(fpr, tpr, label=f"XGBoost {name}",color="dodgerblue", linewidth=4)
    plt.plot(hlsfpr, hlstpr, label=f"{cfg['Precision']} {name}", linestyle="--", color="orange",linewidth=3)
    plt.grid(False)
    plt.legend()
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    hep.cms.text("Phase-2 Simulation Preliminary", fontsize=22)
    hep.cms.lumitext("14 TeV, 200 PU", fontsize=22)
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

#%%
plots.plot_best_pt_roc(
    df_test,
    ptkey="CC_pt",
    score="XGBScore",
    thrs_to_select=[0.85, 0.7, 0.35, 0.3, 0.55, 0.85],
    save=cfg['OutputDir']+"/fig/xgb_pt_roc.pdf",
)


"""

def plot_pt_roc(
    data,
    score="score",
    labkey="label",
    ptkey="CryClu_pt",
    pt_bins=(0, 5, 10, 20, 30, 50, 150),
    save=False,
    lumitext="All Cluster-Track Couples",
    eff=False,
    thrs_to_select=False,
):
    thrs = [0.4, 0.6, 0.8, 0.9]
    colors = ["red", "dodgerblue", "green", "gold"]
    markers = ["o", "s", "^", "v", "X", "*"]

    fig, ax = plt.subplots()
    custom_lines = []
    lines = []
    for idx, (minpt, maxpt) in enumerate(pairwise(pt_bins)):
        pt_data = data[np.bitwise_and(data[ptkey] > minpt, data[ptkey] < maxpt)]

        # ddata=xgb.DMatrix(pt_data[features],label=pt_data["label"],enable_categorical=True)
        preds = pt_data[score].to_numpy()
        y = pt_data[labkey].to_numpy()
        y[y == 2] = 1
        # y=ddata.get_label()

        fpr, tpr, thresh = roc_curve(y, preds)
        # frac=len(pt_data[pt_data["label"]!=1])/len(data[data["label"]!=1])
        # fpr=fpr*frac
        # print(frac)

        if eff:
            if isinstance(eff, Iterable):
                e = eff[idx]
            else:
                e = eff

            score_eff_idx = np.argmin(np.abs(tpr - e))
            score_eff = thresh[score_eff_idx]
            print(f"pt=[{minpt},{maxpt}] GeV:tpr={e}  fpr={fpr[score_eff_idx]:.2f} thr={score_eff:.2f}")
        if thrs_to_select:
            if isinstance(thrs_to_select, Iterable):
                thr_to_select = thrs_to_select[idx]
            else:
                thr_to_select = thrs_to_select
            thr_idx = np.argmin(np.abs(thresh - thr_to_select))
            print(f"pt=[{minpt},{maxpt}] GeV:tpr={tpr[thr_idx]:.2f}  fpr={fpr[thr_idx]:.2f} thr={thr_to_select:.2f}")

        roc_auc = auc(fpr, tpr)
        if maxpt>100:
          maxpt=r"$\infty$"
        pt_range_str=r"$p_T^{\text{Cluster}}$="+f"[{minpt},{maxpt}]"
        auc_str=f"GeV (AUC = {roc_auc:.2f})"
        lab =  f"{pt_range_str} {auc_str}"

        (line,) = ax.plot(fpr, tpr, label=lab, alpha=0.9)
        lines.append(line)
        custom_lines.append(
            Line2D(
                [0],
                [0],
                color=plt.gca().lines[-1].get_color(),
                lw=2,
                linestyle="-",
                label=f"{minpt}-{maxpt} GeV",
                marker=markers[idx],
                markersize=10,
                markeredgecolor="black",
            )
        )
    # ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([-0.05, 0.5])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    line_to_custom_line = {line: custom_line for line, custom_line in zip(lines, custom_lines)}
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [line_to_custom_line.get(handle, handle) for handle in handles]
    ax.legend(handles, labels, fontsize=19)
    hep.cms.text("Phase-2 Simulation Preliminary", ax=ax, fontsize=22)
    hep.cms.lumitext(lumitext, ax=ax, fontsize=22)
    if save:
        fig.savefig(save)
    return ax


def plot_best_pt_roc(
    data,
    score="score",
    labkey="label",
    ptkey="CryClu_pt",
    pt_bins=(0, 5, 10, 20, 30, 50, 150),
    ids=["evId", "CryClu_id"],
    save=False,
    eff=False,
    thrs_to_select=False,
):
    new_data = data.astype(float)
    new_data = new_data.groupby(ids).max(score).reset_index()
    ax = plot_pt_roc(
        new_data,
        score=score,
        labkey=labkey,
        ptkey=ptkey,
        pt_bins=pt_bins,
        save=save,
        lumitext="14 TeV, 200 PU",
        eff=eff,
        thrs_to_select=thrs_to_select,
    )
    ax.grid(False)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1,alpha=0.5,zorder=-10)
    ax.axhline(0.9, linestyle="--", color="black", linewidth=1,alpha=0.5,zorder=-10)
    ax.set_xlabel("Background Efficiency")
    ax.set_ylabel("Signal Efficiency")
    return ax, new_data

"""
