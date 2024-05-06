# %%
import glob
import os
import warnings

import awkward as ak
import dask_awkward as dak
import hist
import uproot
import vector
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from rich import print as pprint

from python.hist_struct import Hist

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", module="coffea.*")
vector.register_awkward()


def sample_generator(dataset_dict: dict, nevents: int | None = None):
    """
    generator that loops over the samples in the dataset and yields a Sample object for each sample

    Args:
    ----
        dataset_dict (_type_): dataset dict parsed from the dataset.yaml file
        nevents (int): number of events to be processed (first nevents events in the sample). Defaults to None.

    Yields:
    ------
        sample: sample object

    """
    base_path = dataset_dict["dataset"]["input_dir"]
    for sample in dataset_dict["samples"]:
        sample_dir = dataset_dict["samples"][sample]["input_sample_dir"]
        path = os.path.join(base_path, sample_dir)
        yield Sample(
            sample,
            tag=dataset_dict["dataset"]["tag"],
            path=path,
            tree_name=dataset_dict["dataset"]["tree_name"],
            scheme_dict=dataset_dict["scheme"],
            nevents=nevents,
        )


class Sample:
    """
    Class that contains the sample information and methods to manipulate it

    Attributes
    ----------
    sample_name : str
        name of the sample
    hist_file : uproot.rootio.ROOTFile
        root file to store the histograms


    """

    def __init__(  # noqa: D417
        self,
        name: str,
        /,
        tag: str = "",
        path: str | None = None,
        tree_name: str | None = None,
        scheme_dict: dict[str, str] | None = None,
        nevents: int | None = None,
        events: NanoEventsFactory | None = None,
    ):
        """
        _summary_

        Args:
        ----
            name (str): name of the sample
            path (str): path of the folder that contains the root files
            tree_name (str): name of the tree
            scheme_dict (Dict[str], optional): Dict {old_name:new_name} to rename collections in the sample. The schemes are defined in the cfg python script. Defaults to None.
            nevents (int, optional): number of events to be processed (first nevents events in the sample). Defaults to None.

        """
        if events is None:
            list_files = glob.glob(os.path.join(path, "*.root"))

            lazy_events = NanoEventsFactory.from_root(
                [{file: tree_name} for file in list_files], schemaclass=NanoAODSchema, delayed=True
            ).events()

            if nevents is None:
                nevents = dak.num(lazy_events, axis=0).compute()

            self.events = ak.Array([{}] * nevents)

            if scheme_dict is None:
                raise ValueError("No scheme provided. Please provide a scheme to rename the collections")
            for old_name, new_name in scheme_dict.items():
                if old_name not in lazy_events.fields:
                    raise ValueError(f"Collection {old_name} does not exist.")
                self.events[new_name] = ak.with_name(lazy_events[old_name][:nevents].compute(), "Momentum4D")
                self.events[new_name].layout.content.parameters["collection_name"] = new_name

        else:
            self.events = events
            nevents = len(events)

        self.nevents = nevents
        self.sample_name = name
        self.tag = tag
        self.hist_file = None
        self.errors={}

    @property
    def fields(self):
        return self.events.fields

    def __len__(self) -> int:
        """
        Return the number of events in the sample

        Returns
        -------
            int: number of event in the sample

        """
        return len(self.events)

    @property
    def n(self) -> int:
        """
        Return the number of events in the sample

        Returns
        -------
            int: number of event in the sample

        """
        return len(self.events)

    def _rename_collection(self, old_name: str, new_name: str) -> None:
        """
        Rename a collection in the sample

        Args:
        ----
            old_name (str): collection to be renamed
            new_name (str): new name of the collection

        """
        if new_name in self.fields:
            raise ValueError(
                f"Collection {new_name} already exists. If you want to override it use __setitem__ method."
            )
        if old_name not in self.fields:
            raise ValueError(f"Collection {old_name} does not exist.")

        idx = self.events.layout._fields.index(old_name)
        self.events.layout._fields[idx] = new_name

    def add_collection(self, collection_name: str, /, ak_array: ak.Array | None = None) -> None:
        """
        Add a new collection. It could be empty or filled with an awkward array

        Args:
        ----
            collection_name (`str`): name of the collection
            ak_array (ak.Array | None, optional): awkward record to add as collection. If None it add an empty collection. Defaults to None.

        """
        if collection_name in self.fields:
            raise ValueError(
                f"Collection {collection_name} already exists. If you want to override it use __setitem__ method."
            )

        if ak_array is None:
            self.events[collection_name] = ak.Array([{}] * self.n, with_name="Momentum4D")
        else:
            self.events[collection_name] = ak.with_name(ak_array, "Momentum4D")

    def get_vars(self, /, collection: str | None = None) -> list[str] | dict[str, list[str]]:
        """
        Returns the list of variables in the sample or in a specific collection

        Args:
        ----
            collection (str | None, optional): name of the collection. If none it returns a dict with all the collections . Defaults to None.

        Returns:
        -------
            list[str] | dict[str,list[str]]: list or dict of list containing all the variables in the sample or in a specific collection

        """
        if collection is None:
            return {field: self.events[field].fields for field in self.fields}
        return self.events[collection].fields

    def create_outfile(self, cfg, path: str) -> None:
        """
        Create a root file to store the histograms

        Args:
        ----
            path (str): path of the folder where the root file will be created

        """
        if self.hist_file is None:
            self.hist_file = uproot.recreate(os.path.join(path, f"{cfg}_{self.sample_name}_{self.tag}.root"))
        else:
            print("Histogram already exists. Did nothing")

    def add_hists(self, hists: list[Hist]) -> None:
        """
        Add histograms to the root file

        Args:
        ----
            hists (list[Hist]): list of Hist object to be created

        """
        if self.hist_file is None:
            raise ValueError("No histogram created. Create one first")
        for h in hists:
            try:
                if h.single_var:
                    to_add=h.add_hist(self.events)
                    self.hist_file[h.name] = to_add

                else:
                    try:
                        if h.collection_name != "":
                            names = h.collection_name.split("/")
                            arr = self.events[*names]
                        else:
                            arr = self.events

                        def recursive(arr, h):
                            if len(arr.fields) > 0:
                                fields = arr.fields
                                for field in fields:
                                    if len(arr[field].fields) > 0:
                                        if h.collection_name != "":
                                            new_name = h.collection_name + "/" + field
                                        else:
                                            new_name = field
                                        new_h = Hist(new_name, hist_range=h.hist_range, bins=h.bins)
                                    else:
                                        new_h = Hist(h.collection_name, field, hist_range=h.hist_range, bins=h.bins)
                                    recursive(arr[field], new_h)
                            #when all the collection is consumed and each variable is deleted, the recursive function will see the empty collection as a variable. Delete it
                            elif h.collection_name=="":
                                    del self.events[*h.var_name.split("/")]
                            else:
                                new_h = Hist(h.collection_name, h.var_name, hist_range=h.hist_range, bins=h.bins)
                                to_add=new_h.add_hist(self.events)
                                self.hist_file[h.name] = to_add

                        recursive(arr, h)
                    except Exception as error:
                        pprint(f"\nError creating hist {h.collection_name}\n")
                        self.errors[f"{h.collection_name}"]=error
                        pprint(error)
            except Exception as error:
                if h.dim == 1:
                    pprint(f"\nError creating hist {h.collection_name}/{h.var_name}\n")
                    self.errors[f"{h.collection_name}/{h.var_name}"]=error
                elif h.dim == 2:
                    pprint(f"\nError creating hist {h.collection_name}/{h.var_name}_vs_{h.collection_name2}/{h.var_name2}\n")
                    self.errors[f"{h.collection_name}/{h.var_name}_vs_{h.collection_name2}/{h.var_name2}"]=error
                print(error)


    def hist_report(self):
        n_errors=len(self.errors)
        if n_errors>0:
            pprint(f"\n{n_errors} errors occurred while creating the histograms")
            for hist in self.errors:
                pprint(f"{hist}: {self.errors[hist]}")
        else:
            pprint("\nAll histograms created successfully")
