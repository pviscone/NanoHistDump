# %%
import glob
import os
import warnings

import awkward as ak
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
            if ".root" not in path:
                list_files = glob.glob(os.path.join(path, "*.root"))
            else:
                list_files = [path]

            dask_events = NanoEventsFactory.from_root(
                [{file: tree_name} for file in list_files],
                schemaclass=NanoAODSchema,
            ).events()

            events_dict = {}
            for old_collection_name, new_collection_name in scheme_dict.items():
                arr = dask_events[old_collection_name]
                if nevents is not None:
                    arr = arr[:nevents]
                arr = ak.with_name(arr.compute(), "Momentum4D")
                arr._layout.content.parameters["collection_name"] = new_collection_name
                events_dict[new_collection_name] = arr

            self.events = ak.Array(events_dict, behavior=dask_events.behavior)
            if nevents is None:
                nevents = len(self.events)

            if scheme_dict is None:
                raise ValueError("No scheme provided. Please provide a scheme to rename the collections")
        else:
            self.events = events
            nevents = len(events)

        self.nevents = nevents
        self.sample_name = name
        self.tag = tag
        self.hist_file = None
        self.errors = {}

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
                to_add = h.add_hist(self.events)
                self.hist_file[h.name] = to_add

            except Exception as error:
                pprint(f"\nError creating hist {h.var_paths}\n")
                self.errors[f"{h.var_paths}"] = error
                print(error)
                # import traceback
                # print(traceback.format_exc())

    def hist_report(self):
        n_errors = len(self.errors)
        if n_errors > 0:
            pprint(f"\n{n_errors} errors occurred while creating the histograms")
            for hist in self.errors:
                pprint(f"{hist}: {self.errors[hist]}")
        else:
            pprint("\nAll histograms created successfully")
