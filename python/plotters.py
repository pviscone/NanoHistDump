import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from hist import intervals
from matplotlib import colors
from rich import print as pprint

hep.styles.cms.CMS["patch.linewidth"] = 2
hep.styles.cms.CMS["lines.linewidth"] = 2
# palette list (10 colors version)
hep.styles.cms.cmap_petroff = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#92dadd",
    "#717581",
]
hep.style.use("CMS")

import sys


def filepath_loader(path_list):
    files_list = []
    for path in path_list:
        if ".root" in path:
            files_list.append(path)
        else:
            files_list.extend(glob.glob(os.path.join(path, "*.root")))
    yield from files_list


class BasePlotter:
    def __init__(
        self,
        name="",
        fig=None,
        ax=None,
        xlim=None,
        ylim=None,
        zlim=None,
        log="",
        grid=True,
        xlabel="",
        ylabel="",
        lumitext="PU 200",
        cmstext="Phase-2 Simulation",
        cmsloc=0,
    ):
        if (fig is None and ax is not None) or (fig is not None and ax is None):
            raise ValueError("If fig is provided, ax must be provided as well, and vice versa.")

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.name = name
        self.lumitext = lumitext
        self.cmstext = cmstext

        hep.cms.text(self.cmstext, ax=self.ax, loc=cmsloc)
        hep.cms.lumitext(self.lumitext, ax=self.ax)
        self.lazy_args = []

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        self.ax.grid(grid)
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.autoscale_view(scalex=True, scaley=True)
        if ylim is not None:
            self.ax.set_ylim(ylim)
            self.ax.autoscale_view(scalex=True, scaley=True)

        if "y" in log.lower():
            self.ax.set_yscale("log")
        if "x" in log.lower():
            self.ax.set_xscale("log")
        if "z" in log.lower():
            self.zlog = True
        else:
            self.zlog = False
        if zlim is None:
            self.zlim = (None, None)

    def add_text(self, *args, **kwargs):
        self.ax.text(*args, **kwargs)

    def save(self, filename, *args, **kwargs):
        pprint(f"Saving {filename}")
        if ".pdf" not in filename:
            raise ValueError("For god's sake, save it as a pdf file!")
        self.fig.savefig(filename, *args, **kwargs)
        plt.close(self.fig)

    def lazy_add(self, to_file, *args,**kwargs):
        self.lazy_args.append((to_file, args, kwargs))

    def lazy_execute(self,file):
        for to_file, args, kwargs in self.lazy_args:
            data=[file[var] for var in to_file]
            self.add(*data, *args, **kwargs)



class TH1(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, hist, **kwargs):
        hep.histplot(hist, ax=self.ax, clip_on=True, **kwargs)
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__

        if kwargs.get("histtype") == "fill":
            self.ax.set_axisbelow(True)


class TH2(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, hist, **kwargs):
        if self.zlog:
            kwargs["norm"] = colors.LogNorm(vmin=self.zlim[0], vmax=self.zlim[1])
        hep.hist2dplot(hist, ax=self.ax, **kwargs)
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__


class TEfficiency(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ylabel="Efficiency", **kwargs)

    def add(self, num, den, **kwargs):
        num = num.to_numpy()
        edges = num[1]
        num = num[0]
        den = den.to_numpy()[0]
        centers = (edges[:-1] + edges[1:]) / 2
        err = np.nan_to_num(intervals.ratio_uncertainty(num, den, "efficiency"), 0)
        eff = np.nan_to_num(num / den, 0)
        self.ax.step(centers, eff, where="mid", **kwargs)
        self.ax.errorbar(centers, eff, yerr=err, fmt="none", color=self.ax.lines[-1].get_color())
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__
