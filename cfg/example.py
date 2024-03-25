from python.hist_struct import Hist


def define(sample):
    pass


#!WARNING: Not specifing the bins and the range force the dask.compute() before the creation of the histogram
hists = [
    # Hist(""),#plot all variables of all collections
    # Hist("GenEl"),  # plot all variables of CryClu collection
    Hist("GenEle"),
    Hist("CryClu"),
    # Hist("CryClu/pt"),#plot pt variable of CryClu collection
    # Hist("CryClu/phi",hist_range=(-3.14,3.14),bins=100),
    # Hist("CryClu/eta_phi",hist_range=[(-5,5),(-3.14,3.14)],bins=[50,50])
]
