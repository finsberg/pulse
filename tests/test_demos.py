import os
import sys
from unittest import mock
import pytest

# There are some problems with some plotting backends in containers.
# Therefore we set the backend here, and execute the files in stead
# of spawning subprocesses.
import matplotlib

matplotlib.use("agg")

curdir = os.path.dirname(os.path.abspath(__file__))
demodir = os.path.join(curdir, "../demo")

demos = [
    (f, root)
    for root, dirname, files in os.walk(demodir)
    for f in files
    if os.path.splitext(f)[-1] == ".py"
]


@pytest.mark.parametrize("filename, root", demos)
def test_demo(filename, root):
    if os.path.basename(root) == "closed_loop":
        return
    os.chdir(root)
    # Add the current folder to sys.path so that
    # python finds the relevant modules
    sys.path.append(root)
    # Execute file
    with mock.patch("pulse.mechanicsproblem.MechanicsProblem.solve") as solve_mock:
        solve_mock.return_value = (1, True)  # (niter, nconv)
        exec(open(filename).read(), globals())
    # Remove the current folder from the sys.path
    sys.path.pop()
