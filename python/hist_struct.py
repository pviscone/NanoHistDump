from itertools import pairwise

import awkward as ak
import hist
import numpy as np
from rich import print as pprint

from python.THaxisUtils import auto_axis, auto_range, split, split_and_flat


class Hist:
    """
    Class to define the histogram to be created. Contains metadata only. If the hist_range and bins are not specified
    they will be computed automatically but the dask.compute() will be called before the creation of the histogram.

    Attributes
    ----------
    name : str
        Name of the hist. It defines the collection and the variable to be plotted. Syntax collection/var for 1D plots
        and collection/var1_var2 for 2D plots.
    hist_range : list[float,float] | list[list[float,float],list[float,float]], optional
        Range of the histogram. Defaults to None (automatic range).
    bins : int | list[int,int], optional
        Binning. Defaults to None (automatic binning).
    entire_sample : bool
        True if the hist is for the entire sample.
    entire_collection : bool
        True if the hist is for the entire collection.
    single_var : bool
        True if the hist is for a single variable.
    collection_name : str
        Name of the collection.
    var_name : str
        Name of the variable.
    dim : int
        Dimension of the histogram (1D or 2D).

    """

    def __init__(
        self,
        var_paths,
        hist_range=None,
        bins=None,
        fill_mode="normal",
        weight=None,
        name=None,
        func=None,
        cut=None,
        **kwargs,
    ):
        """
        Initialize a HistStruct object.

        Args:
        ----
            name (str): Name of the histogram. It defines the collection and the variable to be plotted.
                Syntax collection/var for 1D plots and collection/var1_var2 for 2D plots.
            hist_range (list[float,float] | list[list[float,float],list[float,float]], optional):
                Range of the histogram. Defaults to None (automatic range).
            bins (int | list[int,int], optional): Binning. Defaults to None (automatic binning).

        """
        _fill_modes = ["normal", "rate_vs_ptcut", "rate_pt_vs_score"]
        assert fill_mode in _fill_modes, f"fill_mode must be one of {_fill_modes}"

        self.var_paths = var_paths if isinstance(var_paths, list) else [var_paths]
        self.dim = len(self.var_paths)

        self.hist_range = hist_range if isinstance(hist_range, list) else [hist_range] * self.dim
        self.bins = bins if isinstance(bins, list) else [bins] * self.dim
        self.func = func if isinstance(func, list) else [func] * self.dim
        self.cut = cut if isinstance(cut, list) else [cut] * self.dim

        assert (
            len(self.hist_range) == len(self.var_paths) == len(self.bins) == len(self.func) == len(self.cut) == self.dim
        ), "All lists must have the same length"

        self.delete_on_add_hist = False  #! Never used. To implement
        self.fill_mode = fill_mode
        self.weight = weight
        self.kwargs = kwargs

        self.hist_obj = None

        collections = [var_path.split("~")[0] for var_path in self.var_paths]
        variables = [var_path.split("~")[1] for var_path in self.var_paths]

        if self.dim > 1:
            base_path = "_vs_".join(collections)
            base_path = base_path.replace("/", "_")
            variables = "_vs_".join(variables)
            self.name = f"{base_path}/{variables}"
        else:
            self.name = f"{collections[0]}/{variables[0]}"

        if name is None:
            self.name_from_user = False
        else:
            self.name_from_user = True
            if name[0] == "$":
                self.name = self.name.rsplit("/", 1)[0] + "/" + name[1:]
            else:
                self.name = name

    def add_hist(self, events, verbose=True) -> None:
        if isinstance(self.weight, str):
            path = self.weight.split("/")
            self.weight = events[*path]
        if verbose:
            pprint(f"Creating hist {self.name}")
            pprint(f"var_paths: {self.var_paths}")
            pprint(f"fill_mode: {self.fill_mode}")
            print("\n")
        self.build_hist(events)
        return self.fill(events)

    def _add_ax(self, events, var_path, bins=None, hist_range=None) -> None:
        ax_name = var_path
        if "numpy" in str(type(bins)):
            axis = hist.axis.Variable(bins, name=ax_name)
        elif hist_range is None and bins is None:
            data = split_and_flat(events, ax_name)
            axis = auto_axis(data, self, ax_name)
        elif hist_range is None and bins is not None:
            data = split_and_flat(events, ax_name)
            axis = auto_range(data, self, ax_name)
        elif hist_range is not None and bins is None:
            axis = hist.axis.Regular(50, *hist_range, name=ax_name)
        else:
            axis = hist.axis.Regular(bins, *hist_range, name=ax_name)
        return axis

    def build_hist(self, events):
        axes = [
            self._add_ax(events, var_path, bins=self.bins[idx], hist_range=self.hist_range[idx])
            for idx, var_path in enumerate(self.var_paths)
        ]
        self.hist_obj = hist.Hist(*axes, storage=hist.storage.Weight())

    def fill(self, events):
        n_ev = len(events)
        root_records_keys = [var_path.split("~")[0].split("/")[0] for var_path in self.var_paths]
        leaves_keys = [
            var_path.split("/", 1)[-1] if "/" in var_path else f"~{var_path.split('~')[-1]}"
            for var_path in self.var_paths
        ]

        data = []
        for record_key, leaves_key, cut, func in zip(root_records_keys, leaves_keys, self.cut, self.func):
            root_record = events[record_key]
            if cut is not None:
                root_record = cut(root_record)
            dat = split(root_record, leaves_key)
            if func is not None:
                dat = func(dat)
            data.append(dat)

        if self.fill_mode == "normal":
            # Used for normal histograms
            # Used in 3D for computing genmatch efficiency on objects with a score and WP depending on the online pt (score, genpt, onlinept)
            data = [ak.ravel(d) for d in data]
            weight = ak.drop_none(ak.ravel(self.weight)) if self.weight is not None else 1.0
            self.hist_obj.fill(*data, weight=weight)

        elif self.fill_mode == "rate_vs_ptcut":
            freq_x_bx = 2760.0 * 11246 / 1000
            # Used for computing rate on objects without a score
            # data = [ pt, ...]
            pt = data[0]
            maxpt_mask = ak.argmax(pt, axis=1, keepdims=True)
            additional_data = [ak.drop_none(ak.ravel(array[maxpt_mask])) for array in data[1:]]
            maxpt = ak.drop_none(ak.ravel(pt[maxpt_mask]))
            for thr, pt_bin_center in zip(self.hist_obj.axes[0].edges, self.hist_obj.axes[0].centers):
                self.hist_obj.fill(
                    np.array([pt_bin_center] * int(ak.sum(maxpt >= thr))), *additional_data, weight=freq_x_bx / n_ev
                )
            self.hist_obj.axes[0].label = "Online pT cut"
            if not self.name_from_user:
                self.name = self.name.split("/", 1)[0] + "/rate_vs_ptcut"
                if len(additional_data) > 0:
                    add_vars = [var.split("~")[1] for var in self.var_paths[1:]]
                    self.name += f"+{'_vs_'.join(add_vars)}"

        elif self.fill_mode == "rate_pt_vs_score":
            # Used for computing rate on objects with a score
            # data = [online pt, score , ...]
            freq_x_bx = 2760.0 * 11246 / 1000
            assert self.dim >= 2, "rate_pt_vs_score mode requires at least 2 variables"
            pt = data[0]
            score = data[1]

            score_cuts = self.hist_obj.axes[1].edges[:-1]
            score_centers = self.hist_obj.axes[1].centers
            for score_idx, score_cut in enumerate(score_cuts):
                score_mask = score > score_cut
                maxpt_mask = ak.argmax(pt[score_mask], axis=1, keepdims=True)
                maxpt = ak.ravel(pt[score_mask][maxpt_mask])

                additional_data = [ak.drop_none(ak.ravel(array[score_mask][maxpt_mask])) for array in data[2:]]
                for pt_bin_center, (lowpt, highpt) in zip(
                    self.hist_obj.axes[0].centers, pairwise(self.hist_obj.axes[0].edges)
                ):
                    (
                        self.hist_obj.fill(
                            np.array([pt_bin_center] * int(ak.sum(np.bitwise_and(maxpt >= lowpt, maxpt < highpt)))),
                            np.array(
                                [score_centers[score_idx]] * int(ak.sum(np.bitwise_and(maxpt >= lowpt, maxpt < highpt)))
                            ),
                            *additional_data,
                            weight=freq_x_bx / n_ev,
                        ),
                    )
            self.hist_obj.axes[0].label = "Online pT cut"
            self.hist_obj.axes[1].label = "Score cut"
            if not self.name_from_user:
                self.name = self.name.split("/", 1)[0] + "/rate_pt_vs_score"
                if len(additional_data) > 0:
                    add_vars = [var.split("~")[1] for var in self.var_paths[2:]]
                    self.name += f"+{'_vs_'.join(add_vars)}"

        return self.hist_obj
