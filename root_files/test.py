# %%
import dask_awkward as dak
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import awkward as ak

NanoAODSchema.warn_missing_crossrefs = False

fname = "DoubleElectrons/DoubleElectrons.root"
events = NanoEventsFactory.from_root(
    {fname: "Events"},
    schemaclass=NanoAODSchema,
).events()

dr_cut = 0.2
gen = dak.with_name(events.GenEl, "PtEtaPhiMLorentzVector")
obj1 = dak.with_name(events.DecTkBarrel, "PtEtaPhiMLorentzVector")
obj2 = dak.with_name(events.CaloEGammaCrystalClustersGCT, "PtEtaPhiMLorentzVector")

# gen = gen.compute()
# obj = obj.compute()
obj1= obj1.compute()
obj2 = obj2.compute()
#%%
var=[{"eta": "caloEta", "phi": "caloPhi"}, {"eta": "eta", "phi": "phi"}]
obj1_to_match = obj1
obj2_to_match = obj2
name1="Tk"
name2="CC"
if var is not None:
    obj1_to_match["eta"] = obj1_to_match[var[0]["eta"]]
    obj1_to_match["phi"] = obj1_to_match[var[0]["phi"]]
    obj2_to_match["eta"] = obj2_to_match[var[1]["eta"]]
    obj2_to_match["phi"] = obj2_to_match[var[1]["phi"]]

obj1_to_match = ak.with_name(obj1_to_match, "PtEtaPhiMLorentzVector")
obj2_to_match = ak.with_name(obj2_to_match, "PtEtaPhiMLorentzVector")
n = ak.max(ak.num(obj2_to_match, axis=1))
obj2_to_match = ak.pad_none(obj2_to_match, n)
for i in range(n):
    dr = obj2_to_match[:, i].delta_r(obj1_to_match)
    matched_obj = obj1_to_match[dr < dr_cut]
    argmax = ak.argmax(matched_obj.pt, axis=1, keepdims=True)
    matched_obj = matched_obj[argmax]

    for fields2 in obj2_to_match.fields:
        matched_obj[f"{name2}_{fields2}"] = ak.singletons(obj2_to_match[:, i][fields2])
    for fields1 in obj1_to_match.fields:
        idx = matched_obj.layout.content.fields.index(fields1)
        matched_obj.layout.content.fields[idx] = f"{name1}_{fields1}"

    if i == 0:
        matched_objs = matched_obj
    elif i > 0:
        matched_objs = ak.concatenate([matched_objs, matched_obj], axis=1)




























# %%
obj1 = events.DecTkBarrel
obj2 = events.CaloEGammaCrystalClustersGCT
name1 = obj1.name.split("-")[0]
name2 = obj2.name.split("-")[0]
vars = [{"eta": "caloEta", "phi": "caloPhi"}, {"eta": "eta", "phi": "phi"}]
dr_cut = 0.2
obj1_to_match = obj1
obj2_to_match = obj2

if vars is not None:
    obj1_to_match["eta"] = obj1_to_match[vars[0]["eta"]]
    obj1_to_match["phi"] = obj1_to_match[vars[0]["phi"]]
    obj2_to_match["eta"] = obj2_to_match[vars[1]["eta"]]
    obj2_to_match["phi"] = obj2_to_match[vars[1]["phi"]]

obj1_to_match = dak.with_name(obj1_to_match, "PtEtaPhiMLorentzVector")
obj2_to_match = dak.with_name(obj2_to_match, "PtEtaPhiMLorentzVector")
n = dak.max(dak.num(obj2_to_match, axis=1)).compute()
obj2_to_match = dak.pad_none(obj2_to_match, n)
for i in range(n):
    dr = obj2_to_match[:, i].delta_r(obj1_to_match)
    matched_obj = obj1_to_match[dr < dr_cut]
    argmax = dak.argmax(matched_obj.pt, axis=1, keepdims=True)
    matched_obj = matched_obj[argmax]
    ###########
    for fields2 in obj2_to_match.fields:
        matched_obj[f"{name2}{fields2.capitalize()}"] = dak.singletons(obj2_to_match[:, i])

    if i == 0:
        matched_objs = matched_obj
    elif i > 0:
        matched_objs = dak.concatenate([matched_objs, matched_obj], axis=1)

a = matched_objs.compute()
