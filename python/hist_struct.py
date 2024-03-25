
class Hist:
    def __init__(self, name, hist_range=None, bins=None):
        if name.count("/") > 1:
            raise ValueError("Name should contain at most one '/'")
        if name.count("_") > 2:
            raise ValueError("Name should contain at most two '_' (2d hist)")

        self.name = name
        self.hist_range = hist_range
        self.bins = bins

        self.entire_sample = (self.name == "")
        self.entire_collection = (self.name.count("/") == 0 and self.name!="")
        self.single_var=(self.name.count("/") == 1)
        self.dim = self.name.count("_") + 1

        self.collection_name=None
        self.var_name=None
        if self.entire_collection:
            self.collection_name=self.name.split("/")[0]
        if self.single_var:
            self.collection_name=self.name.split("/")[0]
            self.var_name=self.name.split("/")[1]


