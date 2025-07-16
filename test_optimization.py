import sys
sys.path.append("../")

import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

NUM_ROD_NODES = 11

def main():
    rod = cosserat.CosseratRod(NUM_ROD_NODES, 3.0, cosserat.AnalyticalEllipseCrossSection(0.5, 0.5), 3e6, 0.45)

    tip_force = cosserat.AppliedTipForce( np.array([10000,0,0]), [0,0], True)
    rod.solveOptimizationProblemTorsionalCorrection([tip_force])

if __name__ == "__main__":
    main()
    