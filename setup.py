# System imports
import os
import sys
import platform

from setuptools import setup, find_packages, Command

# Version number
major = 0
minor = 1

on_rtd = os.environ.get('READTHEDOCS') == 'True'


# if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
#     # In the Windows command prompt we can't execute Python scripts
#     # without a .py extension. A solution is to create batch files
#     # that runs the different scripts.
#     batch_files = []
#     for script in scripts:
#         batch_file = script + ".bat"
#         f = open(batch_file, "w")
#         f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
#         f.close()
#         batch_files.append(batch_file)
#     scripts.extend(batch_files)


if on_rtd:
    REQUIREMENTS = []
else:
    REQUIREMENTS = [
        "numpy>=1.13",
        "h5py>=2.5"
    ]

dependency_links = []

setup(name="pulse",
      version="{0}.{1}".format(major, minor),
      description="""
      An cardiac mechanics solver""",
      author="Henrik Finsberg",
      author_email="henriknf@simula.no",
      license="LGPL version 3 or later",
      install_requires=REQUIREMENTS,
      dependency_links=dependency_links,
      packages=['pulse',
                'pulse.material',
                'pulse.example_meshes'],
      package_data={'pulse.example_meshes':  ["*.h5"]},
      package_dir={"pulse": "pulse"},
      )
