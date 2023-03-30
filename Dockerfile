FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-03-01a

COPY . /app
WORKDIR /app

RUN python3 -m pip install "."
