import os
import subprocess as sp
import sys
from pathlib import Path
from unittest import mock

import pytest

try:
    import mshr  # noqa: F401
except ImportError:
    has_mshr = False
else:
    has_mshr = True

# There are some problems with some plotting backends in containers.
# Therefore we set the backend here, and execute the files in stead
# of spawning subprocesses.
import matplotlib

matplotlib.use("agg")

here = Path(__file__).parent
demodir = here.parent.joinpath("demo")


demos = [
    (f, root)
    for root, dirname, files in os.walk(demodir)
    for f in files
    if Path(f).suffix == ".ipynb" and "checkpoint" not in Path(f).name
]


@pytest.mark.parametrize("filename, root", demos)
def test_demo(filename, root):
    if os.path.basename(root) == "closed_loop":
        return
    if os.path.basename(root) == "creating_geometries" and not has_mshr:
        return
    os.chdir(root)
    # Add the current folder to sys.path so that
    # python finds the relevant modules

    sp.check_call(["jupytext", filename, "--to", ".py"])
    py_filename = Path(filename).with_suffix(".py")
    sys.path.append(root)
    # Execute file
    with mock.patch("pulse.mechanicsproblem.MechanicsProblem.solve") as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        exec(open(py_filename).read(), globals())
    # Remove the current folder from the sys.path
    py_filename.unlink()
    sys.path.pop()
