import os
import sys
from pathlib import Path
from shutil import rmtree
from typing import Dict
from typing import List

from setuptools import Command
from setuptools import setup


VERSION = "2021.1.1"

here = Path(__file__).parent


# Load the package's __version__.py module as a dictionary.
about: Dict[str, str] = {}
with open(here.joinpath("pulse/__version__.py")) as f:
    exec(f.read(), about)
about["__version__"] = VERSION


class ReleaseCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options: List[str] = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(here.joinpath("dist"))
        except OSError:
            pass

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push finsberg master --tags")

        sys.exit()


# Where the magic happens:
setup(
    version=about["__version__"],
    cmdclass={"release": ReleaseCommand},
)
