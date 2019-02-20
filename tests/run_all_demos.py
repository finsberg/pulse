import os

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
            exec(open(f).read())
