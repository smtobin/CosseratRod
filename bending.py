import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

from enum import Enum
from multiprocessing import Process

NUM_ROD_NODES = 15
UNDEFORMED_COLOR = [255, 0, 0]
DEFORMED_COLOR = [0, 0, 255]
MODEL_COLOR = [255, 130, 0]
UNDEFORMED_COLOR = [255, 130, 0]
ARROW_COLOR = [255, 0, 0]
FEM_COLOR = [38, 227, 0]
ROD_LENGTH = 2
ROD_WIDTH_X = 1
ROD_WIDTH_Y = 0.5

COMSOL_FOLDER = "./comsol/1x0.5x2block_E=1e5_nu=0.49/"
# COMSOL_UNDEFORMED_FILENAME = COMSOL_FOLDER + "undeformed.stl"
# COMSOL_DEFORMED_FILENAMES = ["deformed_F=500_center_distributed.txt", "deformed_F=500_midpoint.txt", "deformed_F=500_corner.txt"]
# COMSOL_DEFORMED_FILENAMES = ["deformed_F=500_center_distributed.txt"]

SPACING = ROD_WIDTH_X * 2

class FigureType(Enum):
    MODELS = 0
    FEM = 1
    MODELS_AND_FEM = 2

# set the types of figures to create
FIGURE_TYPES = [FigureType.FEM, FigureType.MODELS_AND_FEM]

# Y_FORCES = [0, 7500, 7500, 7500]
# AB_COORDS = [[0,0], [0,0], [0.5,0], [1,0]]

# Y_FORCES = [500, 500, 500]
# AB_COORDS = [[0,0], [0.5, 0.5], [1.0, 1.0]]

Y_FORCES = [500]
AB_COORDS = [[0,0]]

def plotModels(deformed_rods, undeformed_index=0):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")
    plotter.camera.position = [0, -5*ROD_WIDTH_X*len(deformed_rods), ROD_LENGTH]

    # plot each rod
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(len(deformed_rods)-1)/2 + SPACING*i, 0, 0])
        
        for p in mesh.points:
            p += mesh_disp

        if i == undeformed_index:
            mesh_color = UNDEFORMED_COLOR
        else:
            mesh_color = MODEL_COLOR
            # plot force arrow
            # get new tip position where force was applied
            # tip_pos = deformed_rod.tipPosition(AB_COORDS[i])
            # plotter.add_arrows(tip_pos + mesh_disp, np.array([0,1,0]), color=ARROW_COLOR)
        
        plotter.add_mesh(mesh, color=mesh_color, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
        for xsection in deformed_xsections:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=0.7)
       

    plotter.add_floor()
    plotter.show()


def plotFEM(deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    for i,mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        mesh.apply_translation( np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0]) )
        plotter.add_mesh(mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    plotter.add_floor()
    plotter.show()


def plotModelFEM(deformed_rods, deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    num_meshes = len(deformed_rods) + len(deformed_fem_meshes)
    # plot each rod
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(num_meshes-1)/2 + SPACING*i, 0, 0])
        
        for p in mesh.points:
            p += mesh_disp
        
        plotter.add_mesh(mesh, color=MODEL_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
        for xsection in deformed_xsections:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=0.7)

    for i,mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        mesh.apply_translation( np.array([-SPACING*(num_meshes-1)/2 + SPACING*(len(deformed_rods) + i), 0, 0]) )
        plotter.add_mesh(mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    plotter.add_floor()
    plotter.show()

def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
    if FigureType.MODELS in FIGURE_TYPES or FigureType.MODELS_AND_FEM in FIGURE_TYPES:
        rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
        l_rod = cosserat.LinearDeformationCosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
        undeformed_rod = copy.copy(rod)

        deformed_rods = []
        # deformed_rods.append(undeformed_rod)
        for y_force, ab_coords in zip(Y_FORCES, AB_COORDS):
            deformed_rod = copy.copy(rod)
            deformed_rod.solveOptimizationProblem([0,y_force,0], ab_coords)
            deformed_rods.append(deformed_rod)
            # deformed_l_rod = copy.copy(l_rod)
            # deformed_l_rod.solveOptimizationProblem([0,y_force,0], ab_coords)
            # deformed_rods.append(deformed_l_rod)

    ######################################################
    # Load FEM Results
    ######################################################
    print("Loading FEM results...")

    # undeformed_fem_mesh = tm.load_mesh(COMSOL_UNDEFORMED_FILENAME)

    ## HOW TO GENERATE THIS OUTPUT FILE IN COMSOL (because I couldn't figure out how to export the deformed mesh):
    # 1. Run the FEM simulation
    # 2. Under Results, Right click 'Export', then click 'Data'
    # 3. Under 'Dataset', Select the solution
    # 4. Under 'Expressions', add 3 expressions: x+u, y+v, z+w (u, v, w are the x,y,z displacements)
    # 5. Under 'Output', change 'Geometry Level' to 'Surface' (i.e. only print data for surface nodes)
    # 6. Choose a filename and click 'Export' at the top
    # deformed_fem_meshes = []
    # for filename in COMSOL_DEFORMED_FILENAMES:
    #     full_path = COMSOL_FOLDER + filename
    #     loaded_data = np.loadtxt(full_path, comments='%')
    #     print(undeformed_fem_mesh.vertices.shape)
    #     print(loaded_data.shape)
    #     deformed_fem_mesh = utils.getDeformedMeshFromComsolData(full_path, undeformed_fem_mesh)
    #     deformed_fem_meshes.append(deformed_fem_mesh)


    # deformed_fem_meshes.insert(0, undeformed_fem_mesh)

    deformed_fem_meshes = []

    undeformed_mesh_filename = "nastran/undeformed.stl"
    undeformed_nodes_filename = "nastran/orig_nodes.csv"
    node_displacements_filename = "nastran/node_disp.csv"

    ## HOW TO GENERATE THESE OUTPUT FILES IN NASTRAN (because you can't export the deformed mesh)
    # 0. Download FNO Reader (link https://forums.autodesk.com/t5/inventor-nastran-forum/read-binary-results-file-fno-with-a-program/m-p/9020216)
    # 1. In Inventor Nastran, right-click "Results" and click "Show in folder" which will take you to the location of the output files from the analysis
    # 2. Generating .stl File
    #   2a. In FNO Reader, select "NAS to CAD" option
    #   2b. Input the .nas filename with the same name as the analysis output file (should be in the same folder as your .fno output file from step 1)
    #   2c. Hit 'Next' until you get to enter the output filename
    #   2d. Enter output filename and click "Create the Output"
    # 3. Generating the undeformed nodes .csv file
    #   3a. In FNO Reader, select "NAS to Text" option
    #   3b. Input the .nas filename from step 2b, click "Next"
    #   3c. Scroll down until you see the row for "GRID", and check the checkbox next to it
    #   3d. In the drop-down menu at the top, change "All rows" to "Checked rows only", click "Next"
    #   3e. Enter output filename and click "Create the Output"
    # 4. Generating the node displacements .csv file
    #   4a. In FNO Reader, select "FNO to Table" option
    #   4b. Enter the .fno filename from step 1, click "Next"
    #   4c. Under "Number to Output", select all rows (scroll to bottom and Shift+Click to highlight all at once) and click the "<" button
    #   4d. Select "[2] T1 TRANSLATION", "[3] T2 TRANSLATION", and "[4] T3 TRANSLATION" (using Ctrl+Click) and clikc the ">" button. There should just be these 3 outputs on the right side
    #   4e. Click "Next", enter the output filename, and click "Create the Output"
    undeformed_mesh = tm.load_mesh(undeformed_mesh_filename)
    deformed_mesh = utils.getDeformedMeshFromNastranData(undeformed_mesh, undeformed_nodes_filename, node_displacements_filename)
    deformed_fem_meshes.append(deformed_mesh)

    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODELS):
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": deformed_rods}))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, kwargs={"deformed_fem_meshes": deformed_fem_meshes}))
        elif (fig_type == FigureType.MODELS_AND_FEM):
            process_list.append(Process(target=plotModelFEM, kwargs={"deformed_rods": deformed_rods, "deformed_fem_meshes": deformed_fem_meshes}))
        
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()