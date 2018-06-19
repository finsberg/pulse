import os
current = os.path.dirname(os.path.abspath(__file__))
mesh_paths = {'simple_ellipsoid': os.path.join(current, 'simple_ellipsoid.h5'),
              'prolate_ellipsoid': os.path.join(current, 'prolate_ellipsoid.h5'),
              'biv_ellipsoid': os.path.join(current, 'biv_ellipsoid.h5'),
              'benchmark': os.path.join(current, 'benchmark.h5')}

