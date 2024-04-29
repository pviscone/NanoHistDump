# %%
import sys

import awkward as ak
import numpy as np
import pandas as pd
import yaml

sys.path.append("../..")

from cfg.functions.matching import match_obj_to_couple, match_obj_to_obj, match_to_gen, select_match
from cfg.functions.utils import set_name
from python.sample import sample_generator


def parse_yaml(filename):
    with open(filename) as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

save=True
tag="131Xv3"
dataset = parse_yaml(f"../../datasets/{tag}_local.yaml")

labels = {"DoubleElectrons": 1, "MinBias": 0}
dataset["samples"] = {sample: dataset["samples"][sample] for sample in labels}



features = {
    "CryClu": ["standaloneWP","showerShape", "isolation"],
    "Tk": ["hitPattern","nStubs", "chi2Bend", "chi2RPhi", "chi2RZ","isReal"],
    "Couple": ["dEtaCryClu","dPhiCryClu","dPtCryClu","ev_id","label"] #,"pt_weights"
}

Barrel_eta = 1.479

df = pd.DataFrame()
for sample in sample_generator(dataset):
    temp_df = pd.DataFrame()
    print(f"{sample.sample_name=}")
    events = sample.events

    events["CryClu","showerShape"] = events.CryClu.e2x5/events.CryClu.e5x5

    events["ev_id"]=np.arange(len(events))


    if sample.sample_name == "DoubleElectrons":

        events["GenEle"]=events.GenEle[np.abs(events.GenEle.eta)<Barrel_eta]
        events=events[ak.num(events.GenEle)>0]


        events["CryCluGenMatchAll"] = match_to_gen(
            events.CryClu, events.GenEle, etaphi_vars=(("eta", "phi"), ("caloeta", "calophi"))
        )

        events["CryCluGenMatch"] = select_match(events.CryCluGenMatchAll, events.CryCluGenMatchAll.GenEle.idx)
        set_name(events.CryCluGenMatch, "CryCluGenMatch")
        events["TkCryCluGenMatchAll"] = match_obj_to_couple(
            events.Tk, events.CryCluGenMatch, "CryClu", etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
        )
        events["TkCryCluGenMatchAll","ev_id"]=events["ev_id"]
        events["TkCryCluGenMatchAll","label"]=np.ones(len(events))
        ptSignal_array=ak.flatten(ak.drop_none(events["TkCryCluGenMatchAll"].CryCluGenMatch.CryClu.pt))




    if sample.sample_name == "MinBias":
        events=events[:100000]
        events["Tk", "isReal"]=2
        events=events[ak.num(events.GenEle)==0]
        events["TkCryCluGenMatchAll"] = match_obj_to_obj(
            events.Tk, events.CryClu, etaphi_vars=(("caloEta", "caloPhi"), ("eta", "phi"))
        )
        events["TkCryCluGenMatchAll","ev_id"]=events["ev_id"]
        events["TkCryCluGenMatchAll","label"]=np.zeros(len(events))
        events["TkCryCluGenMatchAll","dEtaCryClu"]=events["TkCryCluGenMatchAll","dEta"]
        events["TkCryCluGenMatchAll","dPhiCryClu"]=events["TkCryCluGenMatchAll","dPhi"]
        events["TkCryCluGenMatchAll","dPtCryClu"]=events["TkCryCluGenMatchAll","dPt"]



    if sample.sample_name == "DoubleElectrons":
        cryclu=events["TkCryCluGenMatchAll"].CryCluGenMatch.CryClu
        tk=events["TkCryCluGenMatchAll"].Tk
    elif sample.sample_name == "MinBias":
        cryclu=events["TkCryCluGenMatchAll"].CryClu
        tk=events["TkCryCluGenMatchAll"].Tk

    for collection in features:
        for variable in features[collection]:
            if collection=="CryClu":
                temp_df[f"{collection}_{variable}"]=ak.flatten(ak.drop_none(cryclu[variable]))
            elif collection=="Tk":
                temp_df[f"{collection}_{variable}"]=ak.flatten(ak.drop_none(tk[variable]))
            elif collection=="Couple":
                temp_df[f"{variable}"]=ak.flatten(ak.drop_none(events["TkCryCluGenMatchAll",variable]))

    df=pd.concat([df,temp_df])


#%%

from scipy.interpolate import PchipInterpolator

ptSF=np.load(f"ptSF_{tag}.npy")
interp_func = (PchipInterpolator(ptSF[:,0], np.log(ptSF[:,1])))

def interp(func,x):
    res=np.zeros_like(x)
    res[x>=100]=func(100)
    res[x<100]=func(x[x<100])
    return res

res=np.zeros(len(df))
res[df["label"]==1]=np.exp(interp(interp_func,np.array(ptSignal_array)))
res[df["label"]==0]=1
df["pt_weight"]=res


if save:
    df.to_parquet(f"{tag}.parquet")

# %%
