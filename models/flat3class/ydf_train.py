#%%
import ydf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sys

sys.path.append("/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class")
from utils import plots
hep.styles.use("CMS")

def plot_scores(model,dtrain,dtest,save=False,log=False):
    fig, ax = plt.subplots()
    preds_test = model.predict(dtest)
    preds_train = model.predict(dtrain)
    y_train=dtrain["label"].to_numpy()
    y_test=dtest["label"].to_numpy()

    bins = np.linspace(0, 1, 30)

    classes=np.unique(y_train)
    colors=["dodgerblue","salmon","green"]
    colors=colors[:len(classes)]
    for color,cls in zip(colors,classes):
        cls_pred_test=preds_test[y_test == cls]
        cls_pred_train=preds_train[y_train == cls]
        if cls_pred_train.ndim>1:
            cls_pred_train=1-cls_pred_train[:,0]
            cls_pred_test=1-cls_pred_test[:,0]

        hatch="/"
        marker="v"
        if cls!=0:
            hatch="\\"
            marker="^"

        train=np.histogram(cls_pred_train, bins=bins, density=True)

        centers=(train[1][1:]+train[1][:-1])/2

        ax.plot(centers,train[0],label=f"y={int(cls)} Train",marker=marker,color=color,markersize=10,markeredgecolor="black",zorder=999)

        ax.hist(cls_pred_test, bins=bins, label=f"y={int(cls)} Test", hatch=hatch,density=True,histtype="step",color=color)

    ax.set_xlabel("1-Score(Bkg)")
    if log:
        ax.set_yscale("log")
    ax.legend()
    plt.show()

features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_isSS",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "dEta",
    "dPhi",
    "PtRatio",
    "nMatch",
    #Comment for light model
    "CryClu_isIso",
    "CryClu_isLooseTkIso",
    "CryClu_isLooseTkSS",
    "CryClu_brems",
    "Tk_hitPattern",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RZ",
    #
    #"Tk_pt",
    #"maxPtRatio_other",
    #"minPtRatio_other",
    #"meanPtRatio_other",
    #"stdPtRatio_other",
    ]

filename="/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flat3class/131Xv3.parquet"
df=pd.read_parquet(filename)
#df["dPhi"]=np.abs(df["dPhi"])
#df["dEta"]=np.abs(df["dEta"])


dtrain, dtest = train_test_split(df, test_size=0.1, random_state=666)


model = ydf.GradientBoostedTreesLearner(label="label",
                                    weights="weight",
                                    features=features,
                                    task=ydf.Task.CLASSIFICATION,
                                    num_trees=10,
                                    max_depth=10,
                                    shrinkage=0.6,
                                    loss="MULTINOMIAL_LOG_LIKELIHOOD",
                                    l1_regularization=100.,
                                    l2_regularization=100.,
                                    validation_ratio=0.1,
                                    ).train(dtrain)
model.describe()
#%%

#plot_scores(model,dtrain,dtest)
df["score"]=1-model.predict(df)[:,0]
plots.plot_best_pt_roc(model,df,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85])