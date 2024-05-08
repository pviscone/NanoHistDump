import re

import awkward as ak
import hist
import numpy as np


def auto_axis(data,h):
    bin_edges = np.histogram_bin_edges(data, bins="auto")

    # re madness to do camelcase splitting
    if ("int32" in data.typestr) or (
        re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", h.var_name)).split()[0] == "is"
    ):
        nbin = len(bin_edges) - 1
        length = bin_edges[-1] - bin_edges[0]
        bin_edges = np.linspace(bin_edges[0] - 1, bin_edges[-1] + 1, int((length + 2) * nbin / length))
    return hist.axis.Variable(bin_edges, name=h.var_name)

def auto_range(data,h):
    min_range, max_range = ak.min(data), ak.max(data)
    # re madness to do camelcase splitting
    if ("int32" in data.typestr) or (
        re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", h.var_name)).split()[0] == "is"
    ):
        min_range = min_range - 1
        max_range = max_range + 1
    return hist.axis.Regular(h.bins, min_range, max_range, name=h.var_name)

def split_and_flat(events,collection_name,var_name):
    names = collection_name.split("/")
    data = events[*names][var_name]
    data = ak.drop_none(data)
    if data.ndim > 1:
        data = ak.flatten(data)
    return data

def fill(h,events,fill_mode,weight=None,**kwargs):
    hist_obj=h.hist_obj
    if fill_mode=="normal":
        data=split_and_flat(events,h.collection_name,h.var_name)
        hist_obj.fill(data,weight=weight)

    elif fill_mode=="rate_vs_ptcut":
        n_ev=len(events)
        freq_x_bx=2760.0*11246/1000
        pt=events[*h.collection_name.split("/")][h.var_name]
        for thr,pt_bin_center in zip(hist_obj.axes[0].edges, hist_obj.axes[0].centers):
            hist_obj.fill(pt_bin_center, weight=ak.sum(pt>=thr))

        hist_obj.axes[0].label="Online pT cut"
        h.name=h.name.rsplit("/",2)[0]+"/rate_vs_ptcut"
        hist_obj=hist_obj*freq_x_bx/n_ev


    return hist_obj

