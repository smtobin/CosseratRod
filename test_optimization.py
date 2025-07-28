import sys
sys.path.append("../")

import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

NUM_ROD_NODES = 10

def main():
    rod = cosserat.CosseratRod(NUM_ROD_NODES, 2.0, cosserat.AnalyticalRectCrossSection(1.0, 0.5), 1e5, 0.49)
    cosserat_rod = copy.copy(rod)
    linearized_rod = copy.copy(rod)
    constant_modes_rod = copy.copy(rod)
    linear_modes_rod = cosserat.LinearDeformationCosseratRod(NUM_ROD_NODES, 2.0, cosserat.AnalyticalRectCrossSection(1.0, 0.5), 1e5, 0.49)
    tip_force = cosserat.AppliedTipForce( np.array([0,500,0]), [0,0], True)
    cosserat_rod.solveOptimizationProblemCosserat([tip_force])
    print(f"Cosserat rod tip position: {cosserat_rod.tipPosition()}")
    linearized_rod.solveOptimizationProblemLinearized([tip_force])
    print(f"Linearized constant modes tip position: {linearized_rod.tipPosition()}")
    constant_modes_rod.solveOptimizationProblemTorsionalCorrection([tip_force])
    print(f"Constant modes tip position: {constant_modes_rod.tipPosition()}")
    linear_modes_rod.solveOptimizationProblem([tip_force])
    print(f"Linear modes tip position: {linear_modes_rod.tipPosition()}")

    

if __name__ == "__main__":
    main()
    