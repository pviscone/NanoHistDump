import re

import awkward as ak
import hist
import numpy as np


def auto_axis(data, h, ax_name):
    bin_edges = np.histogram_bin_edges(data, bins="auto")

    # re madness to do camelcase splitting
    if ("int32" in data.typestr) or (
        re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", ax_name)).split()[0] == "is"
    ):
        nbin = len(bin_edges) - 1
        length = bin_edges[-1] - bin_edges[0]
        bin_edges = np.linspace(bin_edges[0] - 1, bin_edges[-1] + 1, int((length + 2) * nbin / length))
    return hist.axis.Variable(bin_edges, name=ax_name)


def auto_range(data, h, ax_name):
    min_range, max_range = ak.min(data), ak.max(data)
    # re madness to do camelcase splitting
    if ("int32" in data.typestr) or (
        re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", ax_name)).split()[0] == "is"
    ):
        min_range = min_range - 1
        max_range = max_range + 1
    return hist.axis.Regular(h.bins, min_range, max_range, name=ax_name)


def split(events, var_path):
    # var_path = dir1/dir2~var[idx]
    idxs=re.findall(r'\[(\d+)\]',var_path)
    idxs = [int(idx) for idx in idxs]
    var_path = var_path.split("[")[0]
    collection_name = var_path.split("~")[0]
    var_name = var_path.split("~")[1]
    if collection_name!="":
        names = collection_name.split("/")
        data = getattr(events[*names],var_name)
    else:
        data = getattr(events,var_name)
    data = data[:,*idxs]
    return ak.drop_none(data)


def split_and_flat(events, var_path):
    # var_path = dir1/dir2~var
    data = split(events, var_path)
    data = ak.ravel(data)
    return data
