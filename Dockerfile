FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16

COPY . /app
WORKDIR /app

RUN python3 -m pip install "."
