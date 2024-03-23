# %%
import glob
import os
from contextlib import suppress

import awkward as ak
import dask_awkward as dak
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

NanoAODSchema.warn_missing_crossrefs = False
tree_name = "Events"


class Sample(dak.lib.core.Array):
    def __init__(self, path, /, scheme_dict=None):
        list_files = glob.glob(os.path.join(path, "*.root"))
        events = NanoEventsFactory.from_root(
            [{file: tree_name} for file in list_files],
            schemaclass=NanoAODSchema,
        ).events()

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
            del self[old_name]

    def __delitem__(self, key):
        self.layout._fields.remove(key)

        all_vars = self.get_vars()
        for collection in all_vars:
            # God knows why it is needed but for the god sake don't touch it
            if collection != key:
                with suppress(Exception):
                    self[collection].layout._content._fields = all_vars[collection]  # noqa: SLF001

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
