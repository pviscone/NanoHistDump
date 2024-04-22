#%%
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import PchipInterpolator

import sys
sys.path.append("../..")

from cfg.functions.matching import match_to_gen, select_match
from python.sample import Sample

hep.style.use("CMS")
bins=np.linspace(0,70,36)
bins=np.concatenate([bins,[120]])

save=False
tag="131Xv3"

signal_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/DoubleElectron"

pu_path="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/MinBias"

schema={"CaloEGammaCrystalClustersGCT":"CryClu",
        "DecTkBarrel":"Tk",
        "GenEl":"GenEle",}

pu=Sample("",path=pu_path,tree_name="Events",scheme_dict=schema).events
signal=Sample("",path=signal_path,tree_name="Events",scheme_dict=schema).events

BarrelEta=1.479

signal["GenEle"]=signal.GenEle[np.abs(signal.GenEle.eta)<BarrelEta]
signal=signal[ak.num(signal.GenEle)>0]


cc_gen=match_to_gen(signal.CryClu,signal.GenEle,etaphi_vars=(("eta","phi"),("caloeta","calophi")))
cc_gen=select_match(cc_gen,cc_gen.GenEle.idx)

centers=(bins[1:]+bins[:-1])/2
#%%
fig,(ax1,ax2)=plt.subplots(2,1)

ptSignal_array=ak.flatten(ak.drop_none(cc_gen.CryClu.pt))
ptSignal=ax1.hist(ptSignal_array, bins=bins,density=True,histtype="step",label="signal")[0]

ptPU_array=ak.flatten(ak.drop_none(pu.CryClu.pt))
ptPU=ax1.hist(ptPU_array, bins=bins,density=True,histtype="step",label="PU")[0]

ax1.legend()
ax1.grid()
ax1.set_yscale("log")

ratio=ptPU/np.mean(ptSignal[1:-1])
smoothed=np.exp(lowess(exog=centers,endog=np.log(1e-18+ratio),frac=0.15,it=20)[:,1])

interp_func = (PchipInterpolator(centers, np.log(smoothed)))
def interp(func,x):
    res=np.zeros_like(x)
    res[x>100]=func(100)
    res[x<100]=func(x[x<100])
    return res

x=np.linspace(0,120,500)
ax2.plot(centers,ratio)
ax2.plot(centers,smoothed,"--",color="red")
ax2.plot(x,np.exp(interp(interp_func,x)),color="goldenrod")
ax2.set_yscale("log")
ax2.grid()
#%%

sf=np.concatenate([x[:,None],np.exp(interp(interp_func,x))[:,None]],axis=1)

if save:
    np.save(f"ptSF_{tag}.npy",sf)
