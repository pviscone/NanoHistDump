import importlib
import os

import typer
import yaml
from rich import print as pprint

from python.sample import sample_generator

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def NanoHistDump(
    config_file: str = typer.Option(..., "-f", "--file", help="specify the python configuration file"),
    dataset_file: str = typer.Option(
        ..., "-i", "--input-dataset", help="specify the yaml file defining the input dataset"
    ),
    samples: str = typer.Option(
        None,
        "-s",
        "--sample",
        help="specify the samples to be processed, separate them by commas if more than one is needed (default all samples)",
    ),
    out_dir: str = typer.Option(None, "-o", "--out_dir", help="override the output directory for the files"),
    nevents: int = typer.Option(None, "-n", "--nevents", help="number of events to process per sample (default all)"),
):
    def parse_yaml(filename):
        with open(filename) as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

    dataset = parse_yaml(dataset_file)

    if out_dir is not None:
        dataset["dataset"]["out_dir"] = out_dir
    os.makedirs(dataset["dataset"]["out_dir"], exist_ok=True)

    if samples is not None:
        samples = samples.split(",")
        if len(set(samples) - set(dataset["samples"])) > 0:
            raise ValueError(
                f"Samples {set(samples)-set(dataset['samples'])} not found in the dataset\nAvailable samples: {dataset['samples'].keys()}"
            )

        dataset["samples"] = {sample: dataset["samples"][sample] for sample in samples}

    cfg = importlib.import_module(config_file.split(".py")[0].replace("/", "."))

    for idx, sample in enumerate(sample_generator(dataset, nevents)):
        pprint(
            f"------------------------- #{idx+1}/{len(dataset['samples'])} {sample.sample_name}-------------------------"
        )
        pprint(f"nevents: {sample.nevents}")
        sample.events=cfg.define(sample.events, sample.sample_name)
        sample.create_outfile(config_file.split("/")[-1].split(".py")[0], dataset["dataset"]["out_dir"])
        sample.add_hists(cfg.hists)
        sample.hist_report()


if __name__ == "__main__":
    app()
    #NanoHistDump(config_file="cfg/new_example.py", dataset_file="datasets/131Xv3.yaml",samples="DoubleElectrons",nevents=1000,out_dir="prova")

