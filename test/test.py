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

fname="/afs/cern.ch/work/p/pviscone/NanoHistDump/root_files/131Xv3/DoubleElectrons_PU200"

scheme = {"CaloEGammaCrystalClustersGCT": "CryClu", "GenEl": "GenEle", "DecTkBarrel": "Tk", "TkEleL2": "TkEle","CaloEGammaCrystalClustersRCT": "CryCluRCT"}

BarrelEta = 1.479

events = Sample("", path=fname, tree_name="Events", scheme_dict=scheme).events

events["GenEle"] = events.GenEle[np.abs(events.GenEle.eta) < BarrelEta]
# events["GenEle"] = events.GenEle[events.GenEle.pt > 5]
events = events[ak.num(events.GenEle) > 0]

#%%


from cfg.functions.matching import delta_phi
from cfg.functions.utils import cartesian

def elliptic_match(obj1,obj2,etaphi_vars,ellipse=None):
    cart, name1, name2 = cartesian(obj1, obj2,nested=True)

    obj1_name=etaphi_vars[0][0].split("/")[:-1]
    obj2_name=etaphi_vars[1][0].split("/")[:-1]
    phi1=cart[name1][*etaphi_vars[0][1].split("/")]
    eta1=cart[name1][*etaphi_vars[0][0].split("/")]
    phi2=cart[name2][*etaphi_vars[1][1].split("/")]
    eta2=cart[name2][*etaphi_vars[1][0].split("/")]

    dphi=delta_phi(phi1,phi2)
    deta=eta1-eta2

    #if ellipse is number
    assert ellipse is not None, "ellipse must be a number or a tuple of pairs of numbers"
    if isinstance(ellipse,int|float):
        mask=(dphi**2/ellipse**2+deta**2/ellipse**2)<1

    elif isinstance(ellipse,tuple|list):
        if isinstance(ellipse[0],int|float) and isinstance(ellipse[1],int|float):
            mask=(dphi**2/ellipse[1]**2+deta**2/ellipse[0]**2)<1
        else:
            mask_arr=[(dphi**2/ellipse_element[1]**2+deta**2/ellipse_element[0]**2)<1 for ellipse_element in ellipse]

            mask=dphi>666
            for elem in mask_arr:
                mask=np.bitwise_or(mask,elem)
    cart=cart[mask]
    cart["dR"]=np.sqrt(dphi[mask]**2+deta[mask]**2)
    cart["dPt"]=cart[name1][*(obj1_name+["pt"])]-cart[name2][*(obj2_name+["pt"])]
    cart["PtRatio"]=cart[name1][*(obj1_name+["pt"])]/cart[name2][*(obj2_name+["pt"])]
    cart["dEta"]=deta[mask]
    cart["dPhi"]=dphi[mask]
    return ak.drop_none(cart)
    #return cart


events["CCTkAll"]=elliptic_match(events.CryClu,events.Tk,etaphi_vars=(("eta","phi"),("caloEta","caloPhi")),ellipse=(0.03,0.3))














# %%
#!ELLIPTIC
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
#!SAME TRACK
def same_track(tk):
    res=[]
    for ev_idx, ev_tk in enumerate(tk):
        matrix=np.array([ev_tk.pt,ev_tk.eta,ev_tk.phi,ev_tk.nStubs,ev_tk.hitPattern]).T
        unique_matrix=np.unique(matrix,axis=0)
        res.append(len(ev_tk)-unique_matrix.shape[0])

    return res
