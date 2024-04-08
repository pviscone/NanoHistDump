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
