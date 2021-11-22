.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

export FENICS_PLOTLY_RENDERER=notebook

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-notebooks ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-notebooks: ## remove notebook checkpoints
	find . -name '*.ipynb_checkpoints' -exec rm -fr {} +

lint: ## check style with flake8
	python3 -m flake8 pulse tests

type: ## Run mypy
	python3 -m mypy pulse tests

test: ## run tests quickly with the default Python
	python3 -m pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source pulse -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pulse.rst
	rm -f docs/pulse.material.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs pulse
	for file in README.md; do \
		cp $$file docs/. ;\
	done
	jupytext demo/benchmark/problem1.py -o docs/problem1.ipynb
	jupytext demo/benchmark/problem2.py -o docs/problem2.ipynb
	jupytext demo/benchmark/problem3.py -o docs/problem3.ipynb
	jupytext demo/biaxial_stress_test/biaxial_stress_test.py -o docs/biaxial_stress_test.ipynb
	jupytext demo/compressible_model/compressible_model.py -o docs/compressible_model.ipynb
	jupytext demo/compute_stress_strain/compute_stress_strain.py -o docs/compute_stress_strain.ipynb
	jupytext demo/custom_geometry/custom_geometry.py -o docs/custom_geometry.ipynb
	jupytext demo/custom_material/demo_custom_material.py -o docs/demo_custom_material.ipynb
	jupytext demo/from_xml/from_xml.py -o docs/from_xml.ipynb
	cp -r demo/from_xml/data docs/.
	jupytext demo/klotz_curve/klotz_curve.py -o docs/klotz_curve.ipynb
	jupytext demo/optimal_control/optimal_control.py -o docs/optimal_control.ipynb
	jupytext demo/rigid_motion/rigid_motion.py -o docs/rigid_motion.ipynb
	jupytext demo/shear_experiment/shear_experiment.py -o docs/shear_experiment.ipynb
	jupytext demo/simple_ellipsoid/simple_ellipsoid.py -o docs/simple_ellipsoid.ipynb
	jupytext demo/unit_cube/unit_cube_demo.py -o docs/unit_cube_demo.ipynb
	jupytext demo/unloading/demo_fixedpointunloader.py -o docs/demo_fixedpointunloader.ipynb
	jupyter-book build docs
	# python -m http.server --directory docs/_build/html

list-demos:
	find ./demo -name '*.ipynb' | xargs jupytext --to py


servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	python3 -m twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} dist/*

dist: clean ## builds source and wheel package
	python setup.py release
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python3 -m pip install --upgrade pip
	python3 -m pip install h5py --no-binary=h5py
	python3 -m pip install -r requirements.txt
	python3 -m pip install .

dev: clean ## Just need to make sure that libfiles remains
	python3 -m pip install -r requirements_dev.txt
	python3 -m pip install -e ".[test,plot,docs,dev]"

bump:
	bump2version patch
