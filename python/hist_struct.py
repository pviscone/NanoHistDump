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
        name,
        hist_range=None,
        bins=None,
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
        if name.count("/") > 1:
            raise ValueError("Name should contain at most one '/'")
        if name.count("_") > 2:
            raise ValueError("Name should contain at most two '_' (2d hist)")

        self.name = name
        self.hist_range = hist_range
        self.bins = bins

        self.entire_sample = self.name == ""
        self.entire_collection = self.name.count("/") == 0 and self.name != ""
        self.single_var = self.name.count("/") == 1
        self.dim = self.name.count("_vs_") + 1

        self.collection_name = None
        self.var_name = None
        if self.entire_collection:
            self.collection_name = self.name.split("/")[0]
        if self.single_var:
            self.collection_name = self.name.split("/")[0]
            self.var_name = self.name.split("/")[1]
