from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from hist import intervals
from matplotlib import colors

hep.styles.cms.CMS["patch.linewidth"]=2
#palette list
# hep.styles.cms.cmap_petroff
hep.style.use("CMS")
#plt.rcParams["axes.axisbelow"] = True


@dataclass
class Text:
    text: str
    x: float
    y: float
    size: int = 12
    color: str = "black"


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
        cmsloc=0
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
        self.zlim = zlim


    def add_text(self,text: Text):
        self.ax.text(text.x, text.y, text.text, size=text.size, color=text.color)

    def save(self, filename, *args, **kwargs):
        if ".pdf" not in filename:
            raise ValueError("For god's sake, save it as a pdf file!")
        self.fig.savefig(filename, *args, **kwargs)


class TH1(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,** kwargs)

    def add(self, hist, **kwargs):
        hep.histplot(hist, ax=self.ax, clip_on=True,**kwargs)
        self.ax.legend()

        if kwargs.get("histtype") == "fill":
            self.ax.set_axisbelow(True)


class TH2(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, hist, **kwargs):
        if self.zlog:
            if self.zlim is None:
                self.zlim = (None, None)
            kwargs["norm"]=colors.LogNorm(vmin=self.zlim[0], vmax=self.zlim[1])
        hep.hist2dplot(hist, ax=self.ax, **kwargs)
        self.ax.legend()

class TEfficiency(BasePlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ylabel="Efficiency", **kwargs)

    def add(self, num, den, **kwargs):
        num=num.to_numpy()
        edges=num[1]
        num=num[0]
        den=den.to_numpy()[0]
        centers = (edges[:-1] + edges[1:]) / 2
        err = np.nan_to_num(intervals.ratio_uncertainty(num, den, "efficiency"), 0)
        eff = np.nan_to_num(num / den, 0)
        self.ax.step(centers, eff, where="mid", **kwargs)
        self.ax.errorbar(centers, eff, yerr=err, fmt="none", color=self.ax.lines[-1].get_color())
        self.ax.legend()


