import os
current = os.path.dirname(os.path.abspath(__file__))
mesh_paths = {'simple_ellipsoid': os.path.join(current, 'simple_ellipsoid.h5'),
              'prolate_ellipsoid': os.path.join(current, 'prolate_ellipsoid.h5'),
              'bechmark': os.path.join(current, 'benchmark.h5')}
data_paths = {'full_data': os.path.join(current, 'full_data.yml'),
              'work_data': os.path.join(current, 'work_data.yml'),
              'unit_data': os.path.join(current, 'unit_data.yml'),
              'biv_data':  os.path.join(current, 'biv_data.yml')}
