import importlib
import os

import typer
import uproot

from python.plotters import TH1, TH2, filepath_loader


def Figurizer(
    config_file: str = typer.Option(..., "-f", "--file", help="specify the python configuration file"),
    inputs: list[str] = typer.Option(  # noqa: B008
        ...,
        "-i",
        "--input",
        help="specify input file containing the histograms or the folder that contains the intup files",
    ),
    out_dir: str = typer.Option(None, "-o", "--out_dir", help="Output folder"),
    all_vars: bool = typer.Option(False, "-a", "--all", help="Plot all histograms"),
):

    plots = importlib.import_module(config_file.split(".py")[0].replace("/", ".")).plots


    for filepath in filepath_loader(inputs):
        file = uproot.open(filepath)
        filename = filepath.rsplit("/", 1)[-1].split(".root")[0]
        os.makedirs(os.path.join(out_dir, filename), exist_ok=True)
        for plot in plots:
            if "/" in plot.name:
                os.makedirs(os.path.join(out_dir, filename, plot.name.rsplit("/", 1)[0]), exist_ok=True)
            plot.lazy_execute(file)
            plot.save(os.path.join(out_dir, filename, plot.name + ".pdf"))

        if all_vars:
            for key in file:
                if "TH" not in str(type(file[key])):
                    continue

                key = key.split(";")[0]  # noqa: PLW2901
                os.makedirs(os.path.join(out_dir, filename, key.rsplit("/", 1)[0]), exist_ok=True)

                if "TH1" in str(type(file[key])):
                    h = TH1(xlabel=key.rsplit("/")[-1])

                elif "TH2" in str(type(file[key])):
                    h = TH2(xlabel=key.rsplit("/")[-1].split("_vs_")[0],
                            ylabel=key.rsplit("/")[-1].split("_vs_")[1],log="z")

                h.add(file[key])
                h.save(os.path.join(out_dir, filename, key + ".pdf"))




if __name__ == "__main__":
    typer.run(Figurizer)
