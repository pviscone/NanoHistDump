import numpy as np
import awkward as ak

BarrelEta = 1.479

def select(events, key, region=None):
    if region.lower() == "barrel":
        events[key] = events.TkEle[np.abs(events[key,"eta"]) < BarrelEta]
    elif region.lower() == "endcap":
        events[key] = events.TkEle[np.bitwise_and(np.abs(events[key,"eta"]) >= BarrelEta, np.abs(events[key,"eta"]) < 2.4)]
    events[key, "hwQual"] = ak.values_astype(events[key].hwQual, np.int32)
    mask_tight_ele = 0b0010
    events[key, "IDTightEle"] = np.bitwise_and(events[key].hwQual, mask_tight_ele) > 0
    events[key] = events.TkEle[events[key,"IDTightEle"]]
    return events
