from glob import glob
from pathlib import Path

import typer
import uproot
from rich import print as pprint


def hadd(target: Path, files: list[Path]):
    if len(files)==1 and "*" in files[0]:
        files = glob(str(files[0])+"/*.root")
    pprint(f"Target: {target}")
    pprint(f"Input Files:\n{files}")
    files=[uproot.open(f) for f in files]

    hist_keys=[key.split(";")[0] for key in files[0].keys() if not isinstance(files[0][key],uproot.reading.ReadOnlyDirectory)]

    outfile=uproot.recreate(target)
    for key in hist_keys:
        hists=[f[key].to_hist() for f in files]
        outfile[key]=sum(hists)
    pprint("Done")


if __name__ == "__main__":
    typer.run(hadd)

