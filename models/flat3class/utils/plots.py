from itertools import pairwise
from collections.abc import Iterable

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import xgboost as xgb
from cycler import cycler
from matplotlib.lines import Line2D
from sklearn.metrics import auc, roc_curve

acab_palette = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

hep.styles.cms.CMS["patch.linewidth"] = 2
hep.styles.cms.CMS["lines.linewidth"] = 2
#hep.styles.cms.CMS["axes.prop_cycle"] = cycler("color", acab_palette)
hep.styles.cms.CMS["figure.autolayout"]=True
hep.styles.cms.CMS["axes.grid"] = True

hep.style.use(hep.style.CMS)

pt_bins=(0,5,10,20,30,50,150)



def plot_loss(eval_result, save=False):
    fig, ax = plt.subplots()
    ax.plot(eval_result["train"]["mlogloss"], label="train")
    ax.plot(eval_result["eval"]["mlogloss"], label="eval")
    #ax.set_yscale("log")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("LogLoss")
    ax.legend()
    plt.show()
    if save:
        fig.savefig(save)
    return ax

def plot_scores(model,dtrain,dtest,save=False,log=False):
    fig, ax = plt.subplots()
    preds_test = model.predict(dtest)
    preds_train = model.predict(dtrain)
    y_train=dtrain.get_label()
    y_test=dtest.get_label()

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
    if save:
        fig.savefig(save)


#!Unused, to remove
def plot_stack_score(model,data,features,save=False,pt_bins=pt_bins):
    preds_sig=[]
    preds_bkg=[]
    for minpt,maxpt in pairwise(pt_bins):
        pt_data=data[np.bitwise_and(data["CryClu_pt"]>minpt,data["CryClu_pt"]<maxpt)]

        pt_data_sig=pt_data[pt_data["label"]==1]
        pt_data_bkg=pt_data[pt_data["label"]==0]
        ddata_sig=xgb.DMatrix(pt_data_sig[features],label=pt_data_sig["label"],enable_categorical=True)
        ddata_bkg=xgb.DMatrix(pt_data_bkg[features],label=pt_data_bkg["label"],enable_categorical=True)
        preds_sig.append(model.predict(ddata_sig))
        preds_bkg.append(model.predict(ddata_bkg))
    fig,ax=plt.subplots()
    ax.hist(preds_bkg,bins=np.linspace(0,1,30),stacked=True,label="PU")
    ax.hist(preds_sig,bins=np.linspace(0,1,30),stacked=True,label="Signal")




def plot_pt_roc(model,data, pt_bins=pt_bins,save=False,lumitext="All Cluster-Track Couples",eff=False):
    thrs=[0.4,0.6,0.8,0.9]
    colors=["red","dodgerblue","green","gold"]
    markers=["o","s","^","v","X","*"]

    fig,ax=plt.subplots()
    custom_lines=[]
    lines=[]
    for idx,(minpt,maxpt) in enumerate(pairwise(pt_bins)):
        pt_data=data[np.bitwise_and(data["CryClu_pt"]>minpt,data["CryClu_pt"]<maxpt)]

        #ddata=xgb.DMatrix(pt_data[features],label=pt_data["label"],enable_categorical=True)
        preds = pt_data["score"].to_numpy()
        y=pt_data["label"].to_numpy()
        y[y==2]=1
        #y=ddata.get_label()

        fpr, tpr, thresh = roc_curve(y, preds)
        #frac=len(pt_data[pt_data["label"]!=1])/len(data[data["label"]!=1])
        #fpr=fpr*frac
        #print(frac)

        if eff:
            if isinstance(eff,Iterable):
                e=eff[idx]
            else:
                e=eff

            score_eff_idx=np.argmin(np.abs(tpr-e))
            score_eff=thresh[score_eff_idx]
            print(f"pt=[{minpt},{maxpt}] GeV:tpr={e}  fpr={fpr[score_eff_idx]:.2f} thr={score_eff:.2f}")
        roc_auc = auc(fpr, tpr)
        lab=f"pt=[{minpt},{maxpt}] (AUC = {roc_auc:.2f}) "
        if eff:
            lab+=f"$\epsilon_S$={e:.2f}: {score_eff:.2f}"

        for col,thr in zip(colors,thrs):
            label=""
            if idx==0:
                label=f"score={thr}"
            thridx=np.argmin(np.abs(thresh-thr))
            ax.plot(fpr[thridx],tpr[thridx],color=col,marker=markers[idx],label=label,markersize=12,markeredgecolor="black",zorder=999)

        line,=ax.plot(fpr, tpr, label=lab,alpha=0.9)
        lines.append(line)
        custom_lines.append(Line2D([0], [0], color=plt.gca().lines[-1].get_color(), lw=2,linestyle="-",label=f"{minpt}-{maxpt} GeV",marker=markers[idx],markersize=10,markeredgecolor="black"))




    #ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([-0.05, 0.5])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    line_to_custom_line = {line: custom_line for line, custom_line in zip(lines, custom_lines)}
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [line_to_custom_line.get(handle, handle) for handle in handles]
    ax.legend(handles, labels,fontsize=19)
    hep.cms.text("Simulation", ax=ax)
    hep.cms.lumitext(lumitext, ax=ax)
    if save:
        fig.savefig(save)
    plt.show()
    return ax


def plot_best_pt_roc(model,data, pt_bins=pt_bins,save=False,eff=False):
    new_data=data.astype(float)
    new_data=new_data.groupby(["evId","CryClu_id"]).max("score").reset_index()
    plot_pt_roc(model,new_data, pt_bins,save=save,lumitext="Best Couple per Cluster",eff=eff)
    return new_data
