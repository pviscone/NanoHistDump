import awkward as ak
import hist
from rich import print as pprint

from python.TH1utils import auto_axis, auto_range, fill
from python.TH2utils import fill2D


class Hist:
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
        collection_name,
        /,
        var_name=None,
        collection_name2=None,
        var_name2=None,
        hist_range=None,
        bins=None,
        fill_mode="normal",
        weight=None,
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
        self.collection_name = collection_name
        self.var_name = var_name
        self.hist_range = hist_range
        self.bins = bins

        self.entire_sample = self.collection_name == ""
        self.collection_name2 = collection_name2
        self.var_name2 = var_name2
        if self.collection_name2 is not None:
            self.dim = 2
        else:
            self.dim = 1

        if self.var_name is None:
            self.entire_collection = True
            self.single_var = False
        else:
            self.entire_collection = False
            self.single_var = True

        self.delete_on_add_hist = True
        self.fill_mode = fill_mode
        self.weight = weight


    def add_hist(self,events) -> None:
        if self.weight is not None:
            path=self.weight.split("/")
            self.weight = events[*path]
        if self.dim == 1:
            pprint(f"Creating hist {self.collection_name}/{self.var_name}")
            hist_obj=self._add_hist_1d(events)
        elif self.dim == 2:
            pprint(f"Creating hist {self.collection_name}/{self.var_name}_vs_{self.collection_name2}/{self.var_name2}")
            hist_obj=self._add_hist_2d(events)
        return hist_obj



    def _add_hist_1d(self, events) -> None:
        names = self.collection_name.split("/")
        data = events[*names][self.var_name]
        if data.ndim > 1:
            data = ak.flatten(data)
        data = ak.drop_none(data)

        if "numpy" in str(type(self.bins)):
            axis = hist.axis.Variable(self.bins, name=self.var_name)
        elif self.hist_range is None and self.bins is None:
            axis=auto_axis(data,self)
        elif self.hist_range is None and self.bins is not None:
            axis=auto_range(data,self)
        elif self.hist_range is not None and self.bins is None:
            axis = hist.axis.Regular(50, *self.hist_range, name=self.var_name)
        else:
            axis = hist.axis.Regular(self.bins, *self.hist_range, name=self.var_name)

        hist_obj = hist.Hist(axis)
        hist_obj=fill(hist_obj,data,self.fill_mode,weight=self.weight)

        if self.delete_on_add_hist:
            del events[*names, self.var_name]

        return hist_obj

    def _add_hist_2d(self, events) -> None:
        var1 = self.var_name
        var2 = self.var_name2
        names1 = self.collection_name.split("/")
        names2 = self.collection_name2.split("/")
        data1 = events[*names1][var1]
        data2 = events[*names2][var2]
        if data1.ndim > 1:
            data1 = ak.flatten(data1)
        if data2.ndim > 1:
            data2 = ak.flatten(data2)
        data1 = ak.drop_none(data1)
        data2 = ak.drop_none(data2)

        if "numpy" in str(type(self.bins[0])):
            axis1= hist.axis.Variable(self.bins[0], name=self.collection_name + "/" + var1)
            axis2= hist.axis.Variable(self.bins[1], name=self.collection_name2 + "/" + var2)
        elif self.hist_range is None and self.bins is None:
            axis1=auto_axis(data1,self)
            axis2=auto_axis(data2,self)
        elif self.hist_range is None and self.bins is not None:
            axis1=auto_range(data1,self)
            axis2=auto_range(data2,self)
        elif self.hist_range is not None and self.bins is None:
            axis1 = hist.axis.Regular(50, *self.hist_range[0], name=self.collection_name + "/" + var1)
            axis2 = hist.axis.Regular(50, *self.hist_range[1], name=self.collection_name2 + "/" + var2)
        else:
            axis1 = hist.axis.Regular(self.bins[0], *self.hist_range[0], name=self.collection_name + "/" + var1)
            axis2 = hist.axis.Regular(self.bins[1], *self.hist_range[1], name=self.collection_name2 + "/" + var2)


        hist_obj = hist.Hist(axis1, axis2)
        return fill2D(hist_obj,data1, data2,self.fill_mode,weight=self.weight)
