# %%
import glob
import os
from contextlib import suppress

import awkward as ak
import dask_awkward as dak
import hist
import hist.dask as hda
import numpy as np
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from python.hist_struct import Hist

NanoAODSchema.warn_missing_crossrefs = False


def sample_generator(dataset_dict, nevents):
    base_path = dataset_dict["dataset"]["input_dir"]
    for sample in dataset_dict["samples"]:
        sample_dir = dataset_dict["samples"][sample]["input_sample_dir"]
        path = os.path.join(base_path, sample_dir)
        yield Sample(sample, path, dataset_dict["dataset"]["tree_name"], dataset_dict["scheme"], nevents=nevents)


class Sample(dak.lib.core.Array):
    def __init__(self, name, path, tree_name, /, scheme_dict=None, nevents=None):
        list_files = glob.glob(os.path.join(path, "*.root"))
        events = NanoEventsFactory.from_root(
            [{file: tree_name} for file in list_files],
            schemaclass=NanoAODSchema,
        ).events()

        if nevents is not None:
            events = events[:nevents]

        self.sample_name = name
        self.hist_file = None

        super().__init__(events.dask, events.name, events._meta, events.divisions)  # noqa: SLF001
        if scheme_dict is not None:
            for old_name, new_name in scheme_dict.items():
                self._rename_collection(old_name, new_name)

    @property
    def __len___(self):
        return dak.num(self, axis=0).compute()

    @property
    def n(self):
        return len(self)

    def _rename_collection(self, old_name, new_name):
        if new_name in self.fields:
            raise ValueError(
                f"Collection {new_name} already exists. If you want to override it use __setitem__ method."
            )
        if old_name not in self.fields:
            raise ValueError(f"Collection {old_name} does not exist.")
        # Try except needed due to the caching mechanism of coffea
        with suppress(Exception):
            self[new_name] = self[old_name]
            # del self[old_name]

    #! Understand how to implement it without breaking everything. Without deleting old collections name, they will be in the outfile if a full sample is requested.
    # Maybe just try to replace the field instead of adding a new and deleting the old one
    """
    def __delitem__(self, key):
        # self.layout._fields.remove(key)

        all_vars = self.get_vars()
        for collection in all_vars:
            # God knows why it is needed but for the god sake don't touch it
            if collection != key:
                with suppress(Exception):
                    self[collection].layout._content._fields = all_vars[collection]  # noqa: SLF001
    """

    def add_collection(self, collection_name, /, ak_array=None):
        if collection_name in self.fields:
            raise ValueError(
                f"Collection {collection_name} already exists. If you want to override it use __setitem__ method."
            )

        if ak_array is None:
            ak_array = ak.Array([{}] * self.n)

        if "dask" not in str(type(ak_array)):
            self[collection_name] = dak.from_awkward(ak_array, self.events.npartitions)
        else:
            self[collection_name] = ak_array

    def get_vars(self, /, collection=None):
        if collection is None:
            return {field: self[field].fields for field in self.fields}
        return self[collection].fields

    def create_outfile(self, path):
        if self.hist_file is None:
            self.hist_file = uproot.recreate(os.path.join(path, f"{self.sample_name}.root"))
        else:
            print("Histogram already exists. Did nothing")

    def add_hists(self, hists):
        if self.hist_file is None:
            raise ValueError("No histogram created. Create one first")
        for h in hists:
            if h.entire_sample:
                for field in self.fields:
                    for var in self[field].fields:
                        name = f"{field}/{var}"
                        var_hist = Hist(name, hist_range=h.hist_range, bins=h.bins)
                        self._add_hists(var_hist)

            elif h.entire_collection:
                for var in self[h.collection_name].fields:
                    var_hist = Hist(f"{h.collection_name}/{var}", hist_range=h.hist_range, bins=h.bins)
                    self._add_hists(var_hist)

            elif h.single_var:
                self._add_hists(h)

    def _add_hists(self, h):
        if h.dim == 1:
            self._add_hists_1d(h)
        elif h.dim == 2:
            self._add_hists_2d(h)

    def _add_hists_1d(self, h):
        data = self[h.collection_name][h.var_name]
        if data.ndim > 1:
            data = dak.flatten(data)
        data = dak.drop_none(data)

        if h.bins is None and h.hist_range is None:
            bin_edges = np.histogram_bin_edges(data.compute(), bins="auto")
            axis = hist.axis.Variable(bin_edges, name=h.var_name)
        elif h.hist_range is None and h.bins is not None:
            min_range, max_range = dak.min(data).compute(), dak.max(data).compute()
            axis = hist.axis.Regular(h.bins, min_range, max_range, name=h.var_name)
        elif h.hist_range is not None and h.bins is None:
            axis = hist.axis.Regular(50, *h.hist_range, name=h.var_name)
        else:
            axis = hist.axis.Regular(h.bins, *h.hist_range, name=h.var_name)

        hist_obj = hda.Hist(axis)
        hist_obj.fill(data)
        self.hist_file[h.name] = hist_obj.compute()

    def _add_hists_2d(self, h):
        var1, var2 = h.var_name.split("_")
        data1 = self[h.collection_name][var1]
        data2 = self[h.collection_name][var2]
        if data1.ndim > 1:
            data1 = dak.flatten(data1)
        if data2.ndim > 1:
            data2 = dak.flatten(data2)
        data1 = dak.drop_none(data1)
        data2 = dak.drop_none(data2)

        if h.bins is None and h.hist_range is None:
            bin_edges1 = np.histogram_bin_edges(data1.compute(), bins="auto")
            axis1 = hist.axis.Variable(bin_edges1, name=var1)
            bin_edges2 = np.histogram_bin_edges(data2.compute(), bins="auto")
            axis2 = hist.axis.Variable(bin_edges2, name=var2)
        elif h.hist_range is None and h.bins is not None:
            min_range1, max_range1 = dak.min(data1).compute(), dak.max(data1).compute()
            axis1 = hist.axis.Regular(h.bins[0], min_range1, max_range1, name=var1)
            min_range2, max_range2 = dak.min(data2).compute(), dak.max(data2).compute()
            axis2 = hist.axis.Regular(h.bins[1], min_range2, max_range2, name=var2)
        elif h.hist_range is not None and h.bins is None:
            axis1 = hist.axis.Regular(50, *h.hist_range[0], name=var1)
            axis2 = hist.axis.Regular(50, *h.hist_range[1], name=var2)
        else:
            axis1 = hist.axis.Regular(h.bins[0], *h.hist_range[0], name=var1)
            axis2 = hist.axis.Regular(h.bins[1], *h.hist_range[1], name=var2)

        hist_obj = hda.Hist(axis1, axis2)
        hist_obj.fill(data1, data2)
        self.hist_file[h.name] = hist_obj.compute()
