import glob
import importlib
import os

import typer
import yaml
from rich import print as pprint

import envs
from python.sample import Sample
from python.scheduler import file_splitting

app = typer.Typer(pretty_exceptions_show_locals=False)


def _run(sample_name, path, dataset_config, schema, nevents, debug, config_file, idx=False):
    sample = Sample(
        sample_name,
        tag=dataset_config["tag"],
        path=path,
        tree_name=dataset_config["tree_name"],
        scheme_dict=schema,
        nevents=nevents,
        debug=debug,
    )
    verbose = not bool(idx)
    if verbose:
        pprint(f"nevents: {sample.nevents}")
    sample.events = cfg.define(sample.events, sample.sample_name)
    sample.create_outfile(config_file.split("/")[-1].split(".py")[0], dataset_config["out_dir"], suffix=idx)
    sample.add_hists(cfg.get_hists(sample.sample_name), verbose=verbose)
    sample.hist_report(verbose=verbose)


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
    collections: str = typer.Option(None, "-c", "--collections", help="collections to be read. separate by commas"),
    debug: bool = typer.Option(False, "-d", "--debug", help="print debug information"),
    ncpu: int = typer.Option(1, "-j", "--ncpu", help="number of cpus to use (-1 to use all available cpus)"),
):
    def parse_yaml(filename):
        with open(filename) as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

    dataset = parse_yaml(dataset_file)
    samples_config = dataset["samples"]
    schema = dataset["scheme"]
    dataset_config = dataset["dataset"]

    if out_dir is not None:
        dataset_config["out_dir"] = out_dir
    os.makedirs(dataset_config["out_dir"], exist_ok=True)

    if samples is not None:
        samples = samples.split(",")
        if len(set(samples) - set(samples_config)) > 0:
            raise ValueError(
                f"Samples {set(samples)-set(samples_config)} not found in the dataset\nAvailable samples: {samples_config.keys()}"
            )

        samples_config = {sample: samples_config[sample] for sample in samples}

    global cfg
    cfg = importlib.import_module(config_file.split(".py")[0].replace("/", "."))

    if collections is not None:
        to_read = collections.split(",")
    else:
        to_read = getattr(cfg, "to_read", None)

    if to_read is not None:
        rev = {value: key for key, value in schema.items()}
        schema = {rev[key]: key for key in to_read}

    ncpu = ncpu if ncpu != -1 else os.cpu_count()
    pprint(f"Running on {ncpu} cpus")

    base_path = dataset_config["input_dir"]
    samples = list(samples_config.keys())
    out_dir = dataset_config["out_dir"]
    for idx, sample_name in enumerate(samples):
        sample_dir = samples_config[sample_name]["input_sample_dir"]
        path = os.path.join(base_path, sample_dir)

        pprint(f"------------------------- #{idx+1}/{len(samples_config)} {sample_name}-------------------------")

        if ncpu > 1:
            files = glob.glob(os.path.join(path, "*.root"))
            tmp_dir = os.path.abspath(os.path.join(out_dir, f"{sample_name}_tmp"))
            dataset_config["out_dir"] = tmp_dir
            os.system(f"rm -rf {tmp_dir}")
            os.makedirs(tmp_dir, exist_ok=True)
            file_splitting(
                _run,
                (sample_name, "file_path", dataset_config, schema, nevents, debug, config_file, "file_idx"),
                files,
                ncpu=ncpu,
            )
            os.system(
                f"hadd -fk {tmp_dir}/../{config_file.split('/')[-1].split('.py')[0]}_{sample_name}_{dataset_config['tag']}.root {tmp_dir}/*.root"
            )
            os.system(f"rm -rf {tmp_dir}")
        else:
            _run(sample_name, path, dataset_config, schema, nevents, debug, config_file)


if __name__ == "__main__":
    envs.set_envs()
    app()
    # NanoHistDump(config_file="cfg/new_example.py", dataset_file="datasets/131Xv3.yaml",samples="DoubleElectrons",nevents=1000,out_dir="prova")
