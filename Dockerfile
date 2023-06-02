FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

COPY . /app
WORKDIR /app

RUN python3 -m pip install "."
