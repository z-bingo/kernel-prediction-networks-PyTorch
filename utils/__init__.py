import sys
import inspect, os

sys.path.insert(0, os.path.join(
                     os.path.dirname(inspect.getfile(inspect.currentframe())),
                     '..'))
sys.path.insert(0, os.path.join(
                     os.path.dirname(inspect.getfile(inspect.currentframe()))))
