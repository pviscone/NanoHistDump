import glob
import os
import sys
from functools import wraps

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from cycler import cycler
from hist import intervals
from matplotlib import colors
from rich import print as pprint

# cms palette list (10 colors version)
petroff10 = [
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

# ACAB (All colorblinds are bastards) plz if you don't know me and you are reading this, I am just joking, these plots are not intended for publication
acab_palette = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

hep.styles.cms.CMS["patch.linewidth"] = 2
hep.styles.cms.CMS["lines.linewidth"] = 2
hep.styles.cms.CMS["axes.prop_cycle"] = cycler("color", acab_palette)

hep.style.use(hep.style.CMS)


def merge_kwargs(**decorator_kwargs):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            kwargs = {**self.kwargs, **kwargs}
            for key, value in decorator_kwargs.items():
                kwargs.setdefault(key, value)
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


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
        **kwargs,
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

        self.kwargs = kwargs

        self.markers = ["v", "^", "X", "P", "d", "*", "p", "o"]
        self.markers_copy = self.markers.copy()

    def add_text(self, *args, **kwargs):
        self.ax.text(*args, **kwargs)

    def add_line(self, x=None, y=None, **kwargs):
        if x is not None and y is None:
            self.ax.axvline(x, **kwargs)
        elif y is not None and x is None:
            self.ax.axhline(y, **kwargs)
        else:
            self.ax.plot(x, y, **kwargs)
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__

    def save(self, filename, *args, **kwargs):
        pprint(f"Saving {filename}")
        if ".pdf" not in filename:
            raise ValueError("For god's sake, save it as a pdf file!")
        self.fig.savefig(filename, *args, **kwargs)
        plt.close(self.fig)

    def lazy_add(self, to_file, mode="normal", *args, **kwargs):
        self.lazy_args.append((to_file, mode, args, kwargs))

    def lazy_execute(self, file):
        for to_file, mode, args, kwargs in self.lazy_args:
            hists_list = [file[var] for var in to_file]
            if mode == "normal":
                self.add(*hists_list, *args, **kwargs)
            elif mode == "rate_vs_pt_score":
                score_cuts = hists_list[0].axes[1].edges[:-1]
                for idx, cut in enumerate(score_cuts):
                    if "cuts" in kwargs:
                        if cut not in kwargs["cuts"]:
                            continue
                    label = kwargs.get("label")
                    if label is not None:
                        label = label.replace("%cut%", cut)
                        kwargs.pop("label")
                    self.add(hists_list[0][:, idx], *args, label=label, **kwargs)
            else:
                raise ValueError(f"mode '{mode}' is not implemented")


class TH1(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @merge_kwargs()
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

    @merge_kwargs()
    def add(self, hist, **kwargs):
        if self.zlog:
            kwargs["norm"] = colors.LogNorm(vmin=self.zlim[0], vmax=self.zlim[1])
        hep.hist2dplot(hist, ax=self.ax, **kwargs)
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__


class TEfficiency(BasePlotter):
    def __init__(self, yerr=True, ylabel="Efficiency", *args, **kwargs):
        super().__init__(*args, ylabel=ylabel, **kwargs)
        self.yerr = yerr

    @merge_kwargs()
    def add(self, num, den, **kwargs):
        num = num.to_numpy()
        edges = num[1]
        num = num[0]
        den = den.to_numpy()[0]
        centers = (edges[:-1] + edges[1:]) / 2
        eff = np.nan_to_num(num / den, 0)
        self.ax.step(centers, eff, where="mid", **kwargs)

        if self.yerr:
            err = np.nan_to_num(intervals.ratio_uncertainty(num, den, "efficiency"), 0)
            self.ax.errorbar(centers, eff, yerr=err, fmt="none", color=self.ax.lines[-1].get_color())

        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__


class TRate(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @merge_kwargs(markeredgecolor="black", markersize=0)
    def add(self, hist, **kwargs):
        centers = hist.axes[0].centers
        values = hist.values()
        if "marker" not in kwargs:
            kwargs["marker"] = self.markers_copy.pop(0)
            if len(self.markers_copy) == 0:
                self.markers_copy = self.markers.copy()

        self.ax.plot(centers, values, **kwargs)

        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__
