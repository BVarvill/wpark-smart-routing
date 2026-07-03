import os
import sys

# The simulation package uses flat intra-package imports (import carpark),
# so tests import it the same way the scripts do: via sys.path.
SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "simulation")
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)
