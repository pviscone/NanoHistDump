# %%
import importlib
import sys

sys.path.append("..")

import awkward as ak
import numpy as np

import python.sample
from cfg.functions.matching import match_to_gen, select_match

importlib.reload(python.sample)
Sample = python.sample.Sample

fname="/data/pviscone/PhD-notes/submodules/NanoHistDump/root_files/131Xv3/DoubleElectron"

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

BarrelEta = 1.479

events = Sample("", path=fname, tree_name="Events", scheme_dict=scheme).events

events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]


# %%
events["CCGenAll"]=match_to_gen(events.CryClu,events.GenEle,etaphi_vars=(("eta","phi"),("caloeta","calophi")))
events["CCGen"]=select_match(events["CCGenAll"],events["CCGenAll"].GenEle.idx)
events["CCGen"]=events["CCGen"][events.CCGen.GenEle.idx==0]


def dphi_eta(obj1,obj2,eta_w,phi_w):
    dphi=(obj1.deltaphi(obj2))*phi_w
    deta=(obj1.deltaeta(obj2))*eta_w
    #return np.sqrt(dphi**2+deta**2)
    return dphi,deta

events["Tk"]=events.Tk[events.Tk.isReal==1]
events["Tk","dphiGen"],events["Tk","detaGen"]=dphi_eta(events.GenEle[:,0],events.Tk,1,1)
events["Tk"]=events.Tk[np.abs(events.Tk.dphiGen)<3]

events["Tk","eta"]=events.Tk.caloEta
events["Tk","phi"]=events.Tk.caloPhi
events["Tk","dphiCC"],events["Tk","detaCC"]=dphi_eta(events.CCGen.CryClu[:,0],events.Tk,1,1)


#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep
mplhep.style.use("CMS")
plt.hist2d(ak.flatten(events["Tk","dphiCC"]).to_numpy(),ak.flatten(events["Tk","detaCC"]).to_numpy(),bins=[50,50],norm = mpl.colors.LogNorm(),range=((-0.3,0.3),(-0.3,0.3)))
plt.xlabel("$ \\Delta \\phi$")
plt.ylabel("$\\Delta \\eta$")

circle=plt.Circle((0,0),radius=0.2,fill=False,color="red")
ax = plt.gca()
ax.add_patch(circle)
plt.grid()





#%%

plt.hist(ak.flatten(events["Tk","dphiCC"]),50,label="$\\Delta  \\phi$")
plt.hist(ak.flatten(events["Tk","detaCC"]),50,label="$\\Delta  \\eta$")
plt.hist(ak.flatten(events["Tk","dphiCC"]),50,histtype="step",color="blue")
plt.yscale("log")
plt.legend()
plt.grid()







# %%

def same_track(tk):
    res=[]
    for ev_idx, ev_tk in enumerate(tk):
        matrix=np.array([ev_tk.pt,ev_tk.eta,ev_tk.phi,ev_tk.nStubs,ev_tk.hitPattern]).T
        unique_matrix=np.unique(matrix,axis=0)
        res.append(len(ev_tk)-unique_matrix.shape[0])

    return res
