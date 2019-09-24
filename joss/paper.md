---
title: 'pulse: A python package based on FEniCS for solving problems in cardiac mechanics'
tags:
  - Python
  - FEniCS
  - Cardiac Mechanics
  - Finite Element Method
  - Continuum Mechanics
authors:
  - name: Henrik Nicolay Topnes Finsberg
    orcid: 0000-0003-3766-2393
    affiliation: "1" # (Multiple affiliations must be quoted)

affiliations:
 - name: Simula Research Laboratory, Oslo, Norway
   index: 1

date: 24 July 2019
bibliography: paper.bib
---
# Summary

`pulse` is a software package based on [FEniCS](https://fenicsproject.org) [@logg2012automated] that aims to solve problems in cardiac mechanics (but is easily extended to solve more general problems in continuum mechanics). `pulse` is a result of the author's PhD thesis [@finsberg2017patient], where most of the relevant background for the code can be found.

While FEniCS offers a general framework for solving PDEs, `pulse` specifically targets problems in continuum mechanics. Therefore, most of the code for applying compatible boundary conditions, formulating the governing equations, choosing appropriate spaces for the solutions and applying iterative strategies, etc., are already implemented, so that the user can focus on the actual problem he/she wants to solve rather than implementing all the necessary code for formulating and solving the underlying equations. 

The user can pick any of the built-in meshes or choose a custom user-defined mesh. The user also need to provide appropriate markers for the boundaries where the boundary conditions will be applied, as well as microstructural information (i.e., information about muscle fiber orientations) if an anisotropic model is to be used. Examples of how to create custom idealized geometries as well as appropriate microstructure can be found in another repository called [ldrb](https://github.com/finsberg/ldrb), which uses the Laplace-Dirichlet Rule-Based (LDRB) algorithm [@bayer2012novel] for assigning myocardial fiber orientations.

Next the user needs to select a material model or create a [custom material model](https://finsberg.github.io/pulse/html/demos/custom_material.html), and define appropriate boundary conditions (Dirichlet, Neumann, or Robin boundary conditions). Finally a `MechanicsProblem` is built using the geometry, material, and boundary conditions. Figure 1 shows the different components involved as well as how they are related.
 
![Visualization of the different components that are part of the `pulse` library.](components.png)

The problem is solved using some iterative strategy, either with an incremental load technique with fixed or adaptive stepping and/or using with a continuation technique [@finsberg2017patient].

It is also possible to estimate the unloaded zero-pressure geometry [@bols2013computational]. This is of particular importance if the geometry being used is taken from a medical image of a patient. In this case, the geometry is subjected to some load due to the blood pressure, and therefore in order to correctly assess the stresses, one need to first find the unloaded geometry.

Papers using this code includes [@finsberg2018estimating] and [@finsberg2018efficient].

A collection of different demos showing how to use the `pulse` library is found in the repository, including an implementation of a [cardiac mechanics bechmark](https://finsberg.github.io/pulse/html/demos/benchmark.html) [@land2015verification], how to use a [custom material model](https://finsberg.github.io/pulse/html/demos/custom_material.html), and how to use a [compressible model](https://finsberg.github.io/pulse/html/demos/compressible_model.html) rather than the default incompressible model.

# References
