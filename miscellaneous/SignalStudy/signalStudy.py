#%%
import sys

sys.path.append("../..")

import awkward as ak
import numpy as np

from cfg.functions.matching import elliptic_match, match_to_gen
from cfg.functions.utils import set_name
from python.sample import Sample

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

signal_name="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU200"

def flat(arr,nested=True):
    if nested:
        arr=ak.flatten(arr,axis=2)
    return ak.flatten(ak.drop_none(arr))

BarrelEta = 1.479
def define(events,ellipse=None):
    #!-------------------TkEle -------------------!#
    events["TkEle"]=events.TkEle[np.abs(events.TkEle.eta)<BarrelEta]
    events["TkEle","hwQual"] = ak.values_astype(events["TkEle"].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events["TkEle","IDTightEle"] = np.bitwise_and(events["TkEle"].hwQual, mask_tight_ele) > 0
    events["TkEle"]=events.TkEle[events.TkEle.IDTightEle]

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


signal_original=Sample("", path=signal_name, tree_name="Events", scheme_dict=scheme).events
#%%
signal = define(signal_original,ellipse=[[0.03,0.3]])
tk=signal.TkCryCluGenMatchAll.Tk
gen=signal.TkCryCluGenMatchAll.CryCluGenMatch.GenEle
cryclu=signal.TkCryCluGenMatchAll.CryCluGenMatch.CryClu
flatTk=flat(tk)
flatGen=flat(gen)
#%%
from python.plotters import TH1
import matplotlib.pyplot as plt
import mplhep
from matplotlib.colors import LogNorm

mplhep.style.use("CMS")

fromGen=flatTk.isReal==1

dVz=flatTk.vz-flatGen.vz
dPt=flatTk.pt-flatGen.pt

def graph(ax,label,log=False):
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    if log:
        ax.set_yscale("log")

fig,ax=plt.subplots()
ax.hist(dVz[fromGen],density=True,bins=50,histtype="step",label="fromGen")
ax.hist(dVz[~fromGen],density=True,bins=50,histtype="step",label="fromPU")
graph(ax,"$\Delta$z0")
plt.savefig("fig/dVz.pdf")

fig,ax=plt.subplots()
ax.hist(dPt[fromGen],density=True,bins=50,histtype="step",label="fromGen",range=(-100,50))
ax.hist(dPt[~fromGen],density=True,bins=50,histtype="step",label="fromPU",range=(-100,50))
graph(ax,"$\Delta p_T$ (Tk-Gen)")
plt.savefig("fig/dPt.pdf")

#%%
mindz0=ak.argmin(np.abs(tk.vz-gen.vz),axis=2,keepdims=True)
mindpt=ak.argmin(np.abs(tk.pt-gen.pt),axis=2,keepdims=True)

tkmindz0=flat(tk[mindz0],nested=True)
tkmindpt=flat(tk[mindpt],nested=True)

plt.hist2d(np.array(tkmindz0.isReal), np.array(tkmindpt.isReal),bins=[[0,1,2,3],[0,1,2,3]],density=True,norm=LogNorm())
plt.colorbar()
plt.xlabel("fromGen of min $\Delta z_0$")
plt.ylabel("fromGen of min $\Delta p_T$")
plt.xticks([0.5,1.5,2.5], [0,1,2])
plt.yticks([0.5,1.5,2.5], [0,1,2])
plt.savefig("fig/fromGen2D.pdf")


plt.figure()
plt.hist(tkmindz0.isReal,bins=[0,1,2,3],density=True,histtype="step",label="fromGen of min $\Delta z_0$")
plt.hist(tkmindpt.isReal,bins=[0,1,2,3],density=True,histtype="step",label="fromGen of min $\Delta p_T$")
plt.legend()
plt.xlabel("fromGen")
plt.xticks([0.5,1.5,2.5], [0,1,2])
plt.grid()
plt.savefig("fig/fromGen1D.pdf")

#%%
fromgen_mask=tk.isReal==1

plt.figure()
plt.hist(ak.flatten(ak.num(tk[fromgen_mask],axis=2)),bins=[1,2,3,4,5],density=True)
plt.xlabel("#fromGen")
plt.grid()
plt.xticks([1.5,2.5,3.5,4.5], [1,2,3,4])
plt.savefig("fig/FromGen_multiplicity.pdf")

#%%

fig,ax=plt.subplots()
for i in range(1,5):
    num_mask=ak.num(tk[fromgen_mask],axis=2)==i
    tk_fromgen=flat(tk[fromgen_mask][num_mask],nested=True)
    gen_fromgen=flat(gen[fromgen_mask][num_mask],nested=True)
    dPt_fromgen=tk_fromgen.pt-gen_fromgen.pt
    ax.hist(dPt_fromgen,density=True,bins=30,histtype="step",label=f"#fromGen: {i}",range=(-80,25),linewidth=3)
    graph(ax,"$\Delta p_T$ (Tk-Gen)")

plt.savefig("fig/dPt_fromGenNum.pdf")

fig,ax=plt.subplots()
for i in range(1,5):
    num_mask=ak.num(tk[fromgen_mask],axis=2)==i
    tk_fromgen=flat(tk[fromgen_mask][num_mask],nested=True)
    gen_fromgen=flat(gen[fromgen_mask][num_mask],nested=True)
    dVz_fromgen=tk_fromgen.vz-gen_fromgen.vz
    ax.hist(dVz_fromgen,density=True,bins=50,histtype="step",label=f"#fromGen: {i}",range=(-5,5),linewidth=3)
    graph(ax,"$\Delta$z0")
plt.savefig("fig/dVz_fromGenNum.pdf")

#%%
fig,ax=plt.subplots()
for i in range(1,5):
    num_mask=ak.num(tk[fromgen_mask],axis=2)==i

    tk_fromgen=tk[fromgen_mask][num_mask]
    gen_fromgen=gen[fromgen_mask][num_mask]
    dPt_fromgen=tk_fromgen.pt-gen_fromgen.pt

    min_mask=ak.argmin(np.abs(dPt_fromgen),axis=2,keepdims=True)

    ax.hist(flat(dPt_fromgen[min_mask],nested=True),density=True,bins=30,histtype="step",label=f"#fromGen: {i}",range=(-80,25),linewidth=3)
    graph(ax,"$min \Delta p_T$ (Tk-Gen)")

plt.savefig("fig/min_dPt_fromGenNum.pdf")
# %%
