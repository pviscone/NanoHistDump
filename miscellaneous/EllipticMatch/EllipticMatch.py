#%%
import importlib
import sys

sys.path.append("../..")

import sys

import awkward as ak
import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

import cfg.functions.matching
import python.sample
from cfg.functions.utils import set_name
from python.plotters import TEfficiency, TRate

mplhep.style.use("CMS")
def flat(arr):
    return np.array(ak.flatten(ak.drop_none(arr)))

def graphics(ax,label):
    ax.set_yscale("log")
    ax.grid()
    ax.set_xlabel(label)
    ax.set_ylabel("Density")

importlib.reload(python.sample)
Sample = python.sample.Sample
importlib.reload(cfg.functions.matching )
elliptic_match, match_to_gen=cfg.functions.matching.elliptic_match, cfg.functions.matching.match_to_gen
scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

noPUname="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU0"
signal_name="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU200"
minbias_name="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/MinBias"

#!#####################DEFINE##############################
BarrelEta = 1.479
def define(events,sample_name=None,ellipse=None):#deta=[0.0125,0.03],dphi=[0.3,0.8]):
    #!-------------------TkEle -------------------!#
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<BarrelEta]
    events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]
    if sample_name == "minbias":
        events = events[ak.num(events.GenEle) == 0]

        events["TkCryCluMatch"] = elliptic_match(
            events.CryClu, events.Tk, etaphi_vars=(("eta", "phi"), ("caloEta", "caloPhi")),ellipse=ellipse)
        mindpt_mask=ak.argmin(np.abs(events["TkCryCluMatch"].dPt),axis=2,keepdims=True)

        events["TkCryCluMatch"]=ak.flatten(events["TkCryCluMatch"][mindpt_mask],axis=2)


        set_name(events.TkCryCluMatch, "TkCryCluMatch")


    else:
        #!-------------------GEN Selection-------------------!#
        events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
        events = events[ak.num(events.GenEle) > 0]

        #!-------------------TkEle-Gen Matching-------------------!#
        events["TkEleGenMatch"] = match_to_gen(
            events.GenEle, events.TkEle, etaphi_vars=(("caloeta", "calophi"), ("caloEta", "caloPhi")),nested=True
        )
        mindpt_mask=ak.argmin(np.abs(events["TkEleGenMatch"].dPt),axis=2,keepdims=True)
        events["TkEleGenMatch"] = ak.flatten(events["TkEleGenMatch"][mindpt_mask],axis=2)


        #!-------------------CryClu-Gen Matching-------------------!#
        events["CryCluGenMatch"] = match_to_gen(
            events.GenEle, events.CryClu, etaphi_vars=(("caloeta", "calophi"), ("eta", "phi")),nested=True)

        mindpt_mask=ak.argmin(np.abs(events["CryCluGenMatch"].dPt),axis=2,keepdims=True)

        events["CryCluGenMatch"]=ak.flatten(events["CryCluGenMatch"][mindpt_mask],axis=2)


        set_name(events.CryCluGenMatch, "CryCluGenMatch")



        #!-------------------Tk-CryClu-Gen Matching-------------------!#

        events["TkCryCluGenMatchAll"] = elliptic_match(events.CryCluGenMatch,
                                                    events.Tk,
                                                    etaphi_vars=[["CryClu/eta", "CryClu/phi"],["caloEta", "caloPhi"]],
                                                    ellipse=ellipse)

        mindpt_mask=ak.argmin(np.abs(events["TkCryCluGenMatchAll"].dPt),axis=2,keepdims=True)
        events["TkCryCluGenMatch"] = ak.flatten(events["TkCryCluGenMatchAll"][mindpt_mask],axis=2)

        set_name(events.TkCryCluGenMatch, "TkCryCluGenMatch")
    return events

#%%
noPU_original = Sample("", path=noPUname, tree_name="Events", scheme_dict=scheme).events
minbias_original=Sample("", path=minbias_name, tree_name="Events", scheme_dict=scheme,nevents=30000).events
signal_original=Sample("", path=signal_name, tree_name="Events", scheme_dict=scheme).events
# %%
#!-------------------etaPhi-------------------!#
noPU=define(noPU_original,ellipse=0.6,sample_name="noPU")
#%%

deta_bins=np.linspace(-0.2,0.2,50)
dphi_bins=np.linspace(-0.8,0.8,80)
dr_bins=np.linspace(0,0.5,80)



fig,ax=plt.subplots(3,1,figsize=(8,18))

ax[0].hist(flat(noPU["TkCryCluGenMatch"].dEta),bins=deta_bins,label="dEta",density=True)
graphics(ax[0],r"$\Delta\eta$ (Tk-Clu)")

ax[1].hist(flat(noPU["TkCryCluGenMatch"].dPhi),bins=dphi_bins,label="dPhi",density=True)
graphics(ax[1],r"$\Delta\phi$ (Tk-Clu)")

ax[2].hist(flat(noPU["TkCryCluGenMatch"].dR),bins=dr_bins,label="dR",density=True)
graphics(ax[2],r"$\Delta R$ (Tk-Clu)")

fig.savefig("MatchFig/etaPhi.pdf")
#%%
pt_bins=[0,10,20,30,50,100]
eta_bins=[0,0.4,0.9,1.4]
#pt_bins=[0,120]
deta_ellipse=0.03
dphi_ellipse=0.3

#deta_ellipse2=0.03
#dphi_ellipse2=0.08
for pt_bins_list in [[0,120],pt_bins]:
    for pt_low,pt_high in zip(pt_bins_list[:-1],pt_bins_list[1:]):
#for eta_bins_list in [[0,BarrelEta],eta_bins]:
    #for eta_low,eta_high in zip(eta_bins_list[:-1],eta_bins_list[1:]):
        pt=flat(noPU["TkCryCluGenMatch"].CryCluGenMatch.CryClu.pt)
        eta=flat(noPU["TkCryCluGenMatch"].CryCluGenMatch.CryClu.eta)
        mask=np.bitwise_and(pt>=pt_low,pt<pt_high)
        #mask=np.bitwise_and(np.abs(eta)>=eta_low,np.abs(eta)<eta_high)
        deta=flat(noPU["TkCryCluGenMatch"].dEta)[mask]
        dphi=flat(noPU["TkCryCluGenMatch"].dPhi)[mask]

        fig,ax=plt.subplots()
        counts, xedges, yedges, im = ax.hist2d(dphi,deta,bins=[dphi_bins,deta_bins],norm=LogNorm(),density=True)
        ax.grid()
        ax.set_xlabel(r"$\Delta\phi$")
        ax.set_ylabel(r"$\Delta\eta$")
        ax.set_title(f"Pt: {pt_low}-{pt_high}")
        fig.colorbar(im, ax=ax)
        ellipse = Ellipse(xy=(0, 0), width=2*dphi_ellipse, height=2*deta_ellipse,
                                edgecolor="r", fc="None", lw=2)
        ax.add_patch(ellipse)

        #ellipse2=Ellipse(xy=(0, 0), width=2*dphi_ellipse2, height=2*deta_ellipse2,edgecolor="r", fc="None", lw=2)
        #ax.add_patch(ellipse2)


        entries=np.sum(mask)
        entries_per_pt=entries/(pt_high-pt_low)
        eta_median=np.median(deta)
        phi_median=np.median(dphi)
        inside_mask=(deta/deta_ellipse)**2+(dphi/dphi_ellipse)**2<1
        #inside_mask2=(deta/deta_ellipse2)**2+(dphi/dphi_ellipse2)**2<1
        #inside_mask=np.bitwise_or(inside_mask1,inside_mask2)
        inside_entries=np.sum(inside_mask)
        inside_ratio=inside_entries/entries



        col_labels = [""]
        row_labels = ["Ellipse1","Entries","Entries inside","Entries/pT_range", "Entries inside/pT_range","Ratio inside/total","Median dEta","Median dPhi"]
        table_vals = [[f"dEta: {deta_ellipse:.2f} dPhi: {dphi_ellipse:.2f}"],
                    #[f"dEta: {deta_ellipse2:.2f} dPhi: {dphi_ellipse2:.2f}"],
                    [entries],
                    [inside_entries],
                    [entries_per_pt],
                    [inside_entries/(pt_high-pt_low)],
                    [f"{inside_ratio*100:.2f}%"],
                    [eta_median],
                    [phi_median]]

        my_table = plt.table(cellText=table_vals,
                            rowLabels=row_labels,
                            colLabels=col_labels,
                            fontsize=320,
                            loc="upper right")
        my_table.scale(0.2,1.8)
        fig.savefig(f"MatchFig/etaphi_{pt_low}_{pt_high}.pdf")


#%%
#!-------------------Efficiencies-------------------!#

signal_ellipse=define(signal_original,ellipse=[[0.03,0.3]],sample_name="signal")
signal_circle=define(signal_original,ellipse=0.2,sample_name="signal")

noPU_ellipse=define(noPU_original,ellipse=[[0.03,0.3]],sample_name="noPU")
noPU_circle=define(noPU_original,ellipse=0.2,sample_name="noPU")

#%%
#!Multiplicity
plt.hist(ak.flatten(ak.num(signal_ellipse.TkCryCluGenMatchAll.Tk.pt,axis=2)),bins=np.linspace(0,10,11),density=True,label="Elliptic Match PU200",histtype="step",linewidth=3)
plt.hist(ak.flatten(ak.num(signal_circle.TkCryCluGenMatchAll.Tk.pt,axis=2)),bins=np.linspace(0,10,11),density=True,label=r"$\Delta R 0.2$ Match PU200",histtype="step",linewidth=3)
plt.hist(ak.flatten(ak.num(noPU_ellipse.TkCryCluGenMatchAll.Tk.pt,axis=2)),bins=np.linspace(0,10,11),density=True,label="Elliptic Match PU0",histtype="step",linewidth=3)
plt.hist(ak.flatten(ak.num(noPU_circle.TkCryCluGenMatchAll.Tk.pt,axis=2)),bins=np.linspace(0,10,11),density=True,label=r"$\Delta R 0.2$ Match PU0",histtype="step",linewidth=3)
plt.grid()
plt.yscale("log")
plt.legend()
plt.xlabel("#Track per cluster")
plt.savefig("MatchFig/multiplicity.pdf")
#%%


ptbins=np.linspace(0,100,51)



tkele_pt=flat(signal_ellipse.TkEleGenMatch.GenEle.pt)
genele_pt=flat(signal_ellipse.GenEle.pt)
TkCryCluMatch_ellipse=flat(signal_ellipse.TkCryCluGenMatch.CryCluGenMatch.GenEle.pt)
TkCryCluMatch_circle=flat(signal_circle.TkCryCluGenMatch.CryCluGenMatch.GenEle.pt)

noPU_cirlce_pt=flat(noPU_circle.TkCryCluGenMatch.CryCluGenMatch.GenEle.pt)
noPU_ellipse_pt=flat(noPU_ellipse.TkCryCluGenMatch.CryCluGenMatch.GenEle.pt)
genele_noPU_pt=flat(noPU_ellipse.GenEle.pt)

noPU_circle_h=hist.Hist(hist.axis.Variable(ptbins))
noPU_circle_h.fill(noPU_cirlce_pt)

noPU_ellipse_h=hist.Hist(hist.axis.Variable(ptbins))
noPU_ellipse_h.fill(noPU_ellipse_pt)

tkele_h=hist.Hist(hist.axis.Variable(ptbins))
tkele_h.fill(tkele_pt)

genele_h=hist.Hist(hist.axis.Variable(ptbins))
genele_h.fill(genele_pt)

genele_noPU_h=hist.Hist(hist.axis.Variable(ptbins))
genele_noPU_h.fill(genele_noPU_pt)

TkCryCluMatch_ellipse_h=hist.Hist(hist.axis.Variable(ptbins))
TkCryCluMatch_ellipse_h.fill(TkCryCluMatch_ellipse)

TkCryCluMatch_circle_h=hist.Hist(hist.axis.Variable(ptbins))
TkCryCluMatch_circle_h.fill(TkCryCluMatch_circle)


#%%
eff=TEfficiency(linewidth=3)
eff.add(tkele_h,genele_h,label="TkEle")
eff.add(TkCryCluMatch_ellipse_h,genele_h,label="Elliptic Match PU200")
eff.add(TkCryCluMatch_circle_h,genele_h,label=r"$\Delta R 0.2$ Match PU200")
eff.add(noPU_ellipse_h,genele_noPU_h,label="Elliptic Match PU0")
eff.add(noPU_circle_h,genele_noPU_h,label=r"$\Delta R 0.2$ Match PU0")

eff.save("MatchFig/eff.pdf")

# %%
#!RATE
minbias_ellipse=define(minbias_original,sample_name="minbias",ellipse=[[0.03,0.3]])
minbias_circle=define(minbias_original,sample_name="minbias",ellipse=0.2)
#%%


def fill_rate(events,h,var):

    n_ev=len(events)
    freq_x_bx=2760.0*11246/1000
    pt=ak.drop_none(events[*var.split("/")])
    maxpt_mask=ak.argmax(pt,axis=1,keepdims=True)
    maxpt=ak.flatten(ak.drop_none(pt[maxpt_mask]))

    for thr,pt_bin_center in zip(h.axes[0].edges, h.axes[0].centers):
        h.fill(pt_bin_center, weight=ak.sum(maxpt>=thr))

    h.axes[0].label="Online pT cut"
    h=h*freq_x_bx/n_ev
    return h


tkEle_rate=hist.Hist(hist.axis.Variable(np.linspace(0,100,100)))


minbias_elliptic_rate=hist.Hist(hist.axis.Variable(np.linspace(0,100,100)))

cryclu_rate=hist.Hist(hist.axis.Variable(np.linspace(0,100,100)))
minbias_circle_rate=hist.Hist(hist.axis.Variable(np.linspace(0,100,100)))


tkEle_rate=fill_rate(minbias_ellipse,tkEle_rate,"TkEle/pt")
cryclu_rate=fill_rate(minbias_ellipse,cryclu_rate,"CryClu/pt")
minbias_elliptic_rate=fill_rate(minbias_ellipse,minbias_elliptic_rate,"TkCryCluMatch/CryClu/pt")
minbias_circle_rate=fill_rate(minbias_circle,minbias_circle_rate,"TkCryCluMatch/CryClu/pt")

rate=TRate(log="y",xlim=(-5,70),markersize=10)
rate.add(tkEle_rate,label="TkEle")
rate.add(cryclu_rate,label="Standalone")
rate.add(minbias_elliptic_rate,label="Elliptic Match")
rate.add(minbias_circle_rate,label=r"$\Delta R 0.2$ Match")
rate.save("MatchFig/rate.pdf")

# %%
