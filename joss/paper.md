---
title: 'pulse: A python package based on FEniCS for solving problems in cardiac mechanics'
tags:
  - Python
  - FEniCS
  - Cardiac Mechanics
  - Finite Element
  - Mechanics
authors:
  - name: Henrik Nicolay 
    orcid: 0000-0003-3766-2393
    affiliation: "1" # (Multiple affiliations must be quoted)

affiliations:
 - name: Simula Research Laboratory, Oslo, Norway
   index: 1

date: 24 May 2019
bibliography: paper.bib
---
# Summary

`pulse` is a software based on FEniCS [@logg2012automated] that can
solve general problems in continuum mechanics but was designed to be
used to model the mechanics of the heart at the organ level.
At this level, the heart, or can be modeled as a hyperelastic and
incompressible continuum body embedded in three dimesional Euclidean
space. However, the user is free to define whatever material model and
impressibility conditions he/she like. 

As a part of the code it there is a quite sophisticated solver that
uses numerical continuation for stepping up the parameters.

It is also implemented an algorithm, known as the backward
displacement method for finding the unloaded stress free geometry.


For a detail background in cardiac mechanics and the methods used in
this code the reader is referred to [@finsberg2017patient].

Papers using this code includes [@finsberg2018estimating] and
[@finsberg2018efficient].

