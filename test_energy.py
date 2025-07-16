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

    rod_z = rod.Z
    
    # edit u1 and u3
    for i in range(0,NUM_ROD_NODES-1):
        rod_z[3*NUM_ROD_NODES + i] = i/100.0
        rod_z[4*NUM_ROD_NODES-1 + i] = i/55.0
        rod_z[5*NUM_ROD_NODES-2 + i] = 1+i/40.0
        rod_z[6*NUM_ROD_NODES-3 + i] = i/100.0
        rod_z[8*NUM_ROD_NODES-5 + i] = i/50.0

    tip_force = cosserat.AppliedTipForce( np.array([0,0,0]), [0,0], True)
    energy = rod._totalEnergy(rod_z, [tip_force])

    print(f"Total energy: {energy}")

    for i in range(0,NUM_ROD_NODES-1):
        rod_z[3*NUM_ROD_NODES + i] += -0.003
        rod_z[4*NUM_ROD_NODES-1 + i] += 0.0005
        rod_z[5*NUM_ROD_NODES-2 + i] += -0.002
        rod_z[6*NUM_ROD_NODES-3 + i] += -0.0006
        rod_z[7*NUM_ROD_NODES-4 + i] += 0.0008
        rod_z[8*NUM_ROD_NODES-5 + i] += 0.001

        # dv1[i] = -0.003;// * i/200.0;
        # dv2[i] = 0.0005;// * i/150.0;
        # dv3[i] = -0.002;// * i/400.0;
        # du1[i] = -0.0006;// * i/150.0;
        # du2[i] = 0.0008;// * i/200.0;
        # du3[i] = 0.001;// * i/250.0;
    energy = rod._totalEnergy(rod_z, [tip_force])
    print(f"New total energy: {energy}")

if __name__ == "__main__":
    main()
    