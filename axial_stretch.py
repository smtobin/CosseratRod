import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

from enum import Enum
from multiprocessing import Process

NUM_ROD_NODES = 11
UNDEFORMED_COLOR = [255, 0, 0]
DEFORMED_COLOR = [0, 0, 255]
MODEL_COLOR = [255, 130, 0]
FEM_COLOR = [38, 227, 0]
ROD_LENGTH = 2
ROD_WIDTH = 1

COMSOL_FOLDER = "./comsol/0.5x2cyl_E=1e5_nu=0.49/"
COMSOL_UNDEFORMED_FILENAME = COMSOL_FOLDER + "undeformed.stl"
COMSOL_DEFORMED_FILENAMES = ["deformed_F=-50000.txt", "deformed_F=-20000.txt", "deformed_F=20000.txt", "deformed_F=50000.txt"]

SPACING = ROD_WIDTH * 2

class FigureType(Enum):
    MODELS = 0
    FEM = 1

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODELS, FigureType.FEM]

# Z_FORCES = [-50000, -25000, -10000, 0, 10000, 25000, 50000]
Z_FORCES = [-50000, -20000, 0, 20000, 50000]
# Z_FORCES = [-10000, 0, 10000]

def plotModels(deformed_rods):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")
    plotter.camera.position = [0, 5*ROD_WIDTH*len(deformed_rods), 2*ROD_LENGTH]

    for i,deformed_rod in enumerate(deformed_rods):
        mesh = deformed_rod.asMesh()
        for p in mesh.points:
            p += np.array([-SPACING*(len(deformed_rods)-1)/2 + SPACING*i, 0, 0])
        
        plotter.add_mesh(mesh, color=MODEL_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    plotter.add_floor()
    plotter.show()


def plotFEM(deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH*len(deformed_fem_meshes), ROD_LENGTH]

    for i,mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        mesh.apply_translation( np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0]) )
        plotter.add_mesh(mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    plotter.add_floor()
    plotter.show()


def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    rod = cosserat.CosseratRod(NUM_ROD_NODES, ROD_LENGTH, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH/2, ROD_WIDTH/2), 1e5, 0.49)
    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(0.5, 0.5), 1e5, 0.49)
    undeformed_rod = copy.copy(rod)

    deformed_rods = []
    for z_force in Z_FORCES:
        deformed_rod = copy.copy(rod)
        deformed_rod.solveOptimizationProblem([0,0,z_force])
        deformed_rods.append(deformed_rod)

    # plotModels(deformed_rods)


    ######################################################
    # Load FEM Results
    ######################################################
    print("Loading FEM results...")

    undeformed_fem_mesh = tm.load_mesh(COMSOL_UNDEFORMED_FILENAME)

    ## HOW TO GENERATE THIS OUTPUT FILE IN COMSOL (because I couldn't figure out how to export the deformed mesh):
    # 1. Run the FEM simulation
    # 2. Under Results, Right click 'Export', then click 'Data'
    # 3. Under 'Dataset', Select the solution
    # 4. Under 'Expressions', add 3 expressions: x+u, y+v, z+w (u, v, w are the x,y,z displacements)
    # 5. Under 'Output', change 'Geometry Level' to 'Surface' (i.e. only print data for surface nodes)
    # 6. Choose a filename and click 'Export' at the top
    deformed_fem_meshes = []
    for filename in COMSOL_DEFORMED_FILENAMES:
        full_path = COMSOL_FOLDER + filename
        loaded_data = np.loadtxt(full_path, comments='%')
        deformed_fem_mesh = utils.getDeformedMeshFromComsolData(loaded_data, undeformed_fem_mesh)
        deformed_fem_meshes.append(deformed_fem_mesh)
    
    # insert the undeformed mesh where F=0
    undeformed_index = np.argwhere(np.array(Z_FORCES) == 0)
    print(undeformed_index)
    if len(undeformed_index) > 0:
        deformed_fem_meshes.insert(undeformed_index.item(), undeformed_fem_mesh)

    
    
    

    # plotFEM(deformed_fem_meshes)
    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODELS):
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": np.array(deformed_rods)} ))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, kwargs={"deformed_fem_meshes": np.array(deformed_fem_meshes)} ))
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()