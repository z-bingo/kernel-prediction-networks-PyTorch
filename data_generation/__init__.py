import sys
import inspect, os

# Need this to import halide. See:
# https://stackoverflow.com/questions/6323860/sibling-package-imports
sys.path.insert(0, os.path.join(
                     os.path.dirname(inspect.getfile(inspect.currentframe())),
                     '..'))
sys.path.insert(0, os.path.join(
                     os.path.dirname(inspect.getfile(inspect.currentframe()))))
