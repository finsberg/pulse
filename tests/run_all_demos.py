import os
import sys
# There are some problems with some plotting backends in containers.
# Therefore we set the backend here, and execute the files in stead
# of spawning subprocesses.
import matplotlib
matplotlib.use("agg")

curdir = os.path.dirname(os.path.abspath(__file__))
demodir = os.path.join(curdir, '../demo')

for root, dirname, files in os.walk(demodir):
    for f in files:

        if os.path.splitext(f)[-1] == '.py':

            if os.path.basename(root) == 'closed_loop':
                continue
            print(root)
            print(f)
            os.chdir(root)
            # Add the current folder to sys.path so that
            # python finds the relevant modules
            sys.path.append(root)
            # Execute file
            exec(open(f).read())
            # Remove the current folder from the sys.path
            sys.path.pop()
