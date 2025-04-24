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
ROD_WIDTH_X = 0.5
ROD_WIDTH_Y = 0.5

COMSOL_FOLDER = "./comsol/1x0.5x2block_E=1e5_nu=0.49/"
COMSOL_UNDEFORMED_FILENAME = COMSOL_FOLDER + "undeformed.stl"
COMSOL_DEFORMED_FILENAME = COMSOL_FOLDER + "deformed_F=500_center.txt"

SPACING = ROD_WIDTH_X * 2

class FigureType(Enum):
    MODELS = 0
    FEM = 1

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODELS, FigureType.FEM]

# Y_FORCES = [0, 7500, 7500, 7500]
# AB_COORDS = [[0,0], [0,0], [0.5,0], [1,0]]

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
        
        plotter.add_mesh(mesh, color=mesh_color, opacity=0.3, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
        for xsection in deformed_xsections:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=0.7)
       

    plotter.add_floor()
    plotter.show()


def plotFEM(undeformed_mesh, deformed_mesh):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")

    deformed_mesh.apply_translation([SPACING,0,0])

    # cross_section, undeformed_cross_section, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH/2)

    plotter.add_mesh(undeformed_mesh, color=[255,0,0], opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)
    plotter.add_mesh(deformed_mesh, color=[0,0,255], opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)
    # plotter.add_mesh(undeformed_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    # plotter.add_mesh(deformed_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_floor(pad=1)

    # tip_cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH - 1e-5)

    # plotter.add_text(f"Undeformed tip position: {tip_cross_section.centroid(undeformed_mesh.vertices)}\nDeformed tip position:   {tip_cross_section.centroid(deformed_mesh.vertices)}", position='lower_left', font_size=14)

    plotter.show()


def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
    rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
    l_rod = cosserat.LinearDeformationCosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
    undeformed_rod = copy.copy(rod)

    deformed_rods = []
    deformed_rods.append(undeformed_rod)
    for y_force, ab_coords in zip(Y_FORCES, AB_COORDS):
        deformed_rod = copy.copy(rod)
        deformed_rod.solveOptimizationProblem([0,y_force,0], ab_coords)
        deformed_rods.append(deformed_rod)
        deformed_l_rod = copy.copy(l_rod)
        deformed_l_rod.solveOptimizationProblem([0,y_force,0], ab_coords)
        deformed_rods.append(deformed_l_rod)


    ######################################################
    # Load FEM Results
    ######################################################
    print("Lodaing FEM results...")

    ## HOW TO GENERATE THIS OUTPUT FILE IN COMSOL (because I couldn't figure out how to export the deformed mesh):
    # 1. Run the FEM simulation
    # 2. Under Results, Right click 'Export', then click 'Data'
    # 3. Under 'Dataset', Select the solution
    # 4. Under 'Expressions', add 3 expressions: x+u, y+v, z+w (u, v, w are the x,y,z displacements)
    # 5. Under 'Output', change 'Geometry Level' to 'Surface' (i.e. only print data for surface nodes)
    # 6. Choose a filename and click 'Export' at the top
    loaded_data = np.loadtxt(COMSOL_DEFORMED_FILENAME, comments='%')
    

    undeformed_mesh = tm.load_mesh(COMSOL_UNDEFORMED_FILENAME)
    deformed_mesh = utils.getDeformedMeshFromComsolData(loaded_data, undeformed_mesh)


    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODELS):
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": deformed_rods}))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, args=(undeformed_mesh, deformed_mesh)))
        
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()