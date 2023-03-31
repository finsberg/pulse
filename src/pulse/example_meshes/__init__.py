import os

current = os.path.dirname(os.path.abspath(__file__))


def join(p):
    return os.path.join(current, p)


mesh_paths = {
    "simple_ellipsoid": join("simple_ellipsoid.h5"),
    "prolate_ellipsoid": join("prolate_ellipsoid.h5"),
    "biv_ellipsoid": join("biv_ellipsoid.h5"),
    "benchmark": join("benchmark.h5"),
    "ellipsoid": join("ellipsoid.h5"),
}
