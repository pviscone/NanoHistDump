import glob
import os
import sys
from collections.abc import Iterable
from functools import wraps
from itertools import pairwise
from numbers import Number

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from cycler import cycler
from hist import Hist, intervals, loc, storage
from hist import rebin as Rebin
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
hep.styles.cms.CMS["legend.frameon"] = True
hep.styles.cms.CMS["figure.autolayout"] = True
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

def convert_to_hist(*hists):
    if len(hists)>1:
        res=[]
        for h in hists:
            if not isinstance(h, Hist):
                res.append(h.to_hist())
            else:
                return res.append(h)
        return res
    if not isinstance(hists[0], Hist):
        return hists[0].to_hist()
    return hists[0]

def set_palette(palette):
    hep.styles.cms.CMS["axes.prop_cycle"] = cycler("color", palette)
    hep.style.use(hep.style.CMS)

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
        rebin=1,
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
        self.rebin = Rebin(rebin)

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
        hist=convert_to_hist(hist)
        hist = hist[self.rebin]
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
        hist=convert_to_hist(hist)
        hist = hist[self.rebin]
        if self.zlog:
            kwargs["norm"] = colors.LogNorm(vmin=self.zlim[0], vmax=self.zlim[1])
        hep.hist2dplot(hist, ax=self.ax, **kwargs)
        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__


class TEfficiency(BasePlotter):
    def __init__(self, yerr=True, xerr=False, ylabel="Efficiency", step=True, fillerr=False, *args, **kwargs):
        super().__init__(*args, ylabel=ylabel, **kwargs)
        self.yerr = yerr
        self.xerr = xerr
        self.step = step
        self.fillerr = fillerr

    @merge_kwargs(linewidth=3, markeredgecolor="Black", markersize=0, errcapsize=3, errlinewidth=2, errzorder=-99, fillalpha=0.3)
    def add(self, num, den, **kwargs):
        keys=list(kwargs.keys())
        err_kwargs = {key.split("err")[1]:kwargs.pop(key) for key in keys if key.startswith("err")}
        fill_kwargs = {key.split("fill")[1]:kwargs.pop(key) for key in keys if key.startswith("fill")}
        num = convert_to_hist(num)[self.rebin]
        den = convert_to_hist(den)[self.rebin]
        num = num.to_numpy()
        edges = num[1]
        num = num[0]
        den = den.to_numpy()[0]
        centers = (edges[:-1] + edges[1:]) / 2
        eff = np.nan_to_num(num / den, 0)
        if "marker" not in kwargs:
            kwargs["marker"] = self.markers_copy.pop(0)
            if len(self.markers_copy) == 0:
                self.markers_copy = self.markers.copy()
        if self.step:
            self.ax.step(centers, eff, where="mid", **kwargs)
        else:
            self.ax.plot(centers, eff, **kwargs)

        if self.yerr:
            err = np.nan_to_num(intervals.ratio_uncertainty(num, den, "efficiency"), 0)
            if self.fillerr:
                self.ax.fill_between(centers, eff-err[0], eff+err[1], color=self.ax.lines[-1].get_color(),**fill_kwargs)
            else:
                xerr = np.diff(edges) / 2 if (not self.step and self.xerr) else None
                self.ax.errorbar(centers, eff, yerr=err, xerr=xerr, fmt="none", color=self.ax.lines[-1].get_color(), **err_kwargs)

        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__

    @merge_kwargs()
    def add_scoreCuts(self, numhist3d, den, ptedges_thr, **kwargs):
        if isinstance(ptedges_thr, Iterable):
            numhist3d=convert_to_hist(numhist3d)
            den=convert_to_hist(den)
            thr_list = ptedges_thr[1]
            pt_edges = ptedges_thr[0]
            hist = Hist(numhist3d.axes[1],storage=storage.Weight())
            for thr, (minpt, maxpt) in zip(thr_list, pairwise(pt_edges)):
                integrated = numhist3d.integrate(2, loc(minpt), loc(maxpt))
                temp_h = integrated.integrate(0, loc(thr), None)
                hist += temp_h
                self.ax.axvline(minpt, color="red", linestyle="--", linewidth=1.25, zorder=-2, alpha=0.6)
        elif isinstance(ptedges_thr, Number):
            hist = numhist3d.integrate(2).integrate(0, loc(ptedges_thr), None)

        return self.add(hist, den, **kwargs)


class TRate(BasePlotter):
    def __init__(self, *args, yerr=True, fillerr=False, log="y", ylabel="Rate [kHz]", **kwargs):
        super().__init__(*args, log=log, ylabel=ylabel, **kwargs)
        self.freq_x_bx = 2760.0 * 11246 / 1000
        self.yerr = yerr
        self.fillerr = fillerr

    @merge_kwargs(markeredgecolor="black", markersize=7, linewidth=3, errcapsize=2, errlinewidth=1, errzorder=-99, fillalpha=0.3)
    def add(self, hist, **kwargs):
        keys=list(kwargs.keys())
        err_kwargs = {key.split("err")[1]:kwargs.pop(key) for key in keys if key.startswith("err")}
        fill_kwargs = {key.split("fill")[1]:kwargs.pop(key) for key in keys if key.startswith("fill")}
        hist = convert_to_hist(hist)[self.rebin]
        centers = hist.axes[0].centers
        values = hist.values()
        variances = hist.variances()
        if "marker" not in kwargs:
            kwargs["marker"] = self.markers_copy.pop(0)
            if len(self.markers_copy) == 0:
                self.markers_copy = self.markers.copy()

        n_ev=(values/variances)[0]*self.freq_x_bx
        n_bin=(values**2/variances)
        rate=values
        self.ax.plot(centers, rate, **kwargs)
        if self.yerr:
            err = np.nan_to_num(intervals.ratio_uncertainty(n_bin, n_ev, "efficiency"), 0)*self.freq_x_bx
            if self.fillerr:
                self.ax.fill_between(centers, rate-err[0], rate+err[1], color=self.ax.lines[-1].get_color(),**fill_kwargs)
            else:
                xerr = np.diff(hist.axes[0].edges) / 2
                self.ax.errorbar(centers, rate, yerr=err, xerr=xerr, fmt="none", color=self.ax.lines[-1].get_color(), **err_kwargs)

        sys.stderr = open(os.devnull, "w")
        self.ax.legend()
        sys.stderr = sys.__stderr__

    @merge_kwargs(markeredgecolor="black", markersize=10)
    def add_scoreCuts(self, hist2d, ptedges_thr, **kwargs):
        hist2d=convert_to_hist(hist2d)
        if ptedges_thr is not None:
            if isinstance(ptedges_thr, Iterable):
                pt_edges = ptedges_thr[0]
                thrs = ptedges_thr[1]
                hist = Hist(hist2d.axes[0], storage=storage.Weight())
                for thr, (minpt, maxpt) in zip(thrs, pairwise(pt_edges)):
                    mask = np.ones_like(hist2d.axes[0].centers)
                    idx_mask = np.bitwise_and(hist2d.axes[0].centers > minpt, hist2d.axes[0].centers < maxpt)
                    mask[~idx_mask] = 0
                    hist += hist2d[:, loc(thr)] * mask
                    self.ax.axvline(minpt, color="red", linestyle="--", linewidth=1, alpha=0.8, zorder=-2)
            elif isinstance(ptedges_thr, Number):
                hist = hist2d[:, loc(ptedges_thr)]

        rate_hist = Hist(hist.axes[0], storage=storage.Weight())
        for i in range(len(hist.values())):
            rate_hist[i] = hist.integrate(0, i, None)

        return self.add(rate_hist, **kwargs)

#ADD THStack, THRatio, THPull, THResiduals