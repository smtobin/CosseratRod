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
ROD_LENGTH = 2
ROD_WIDTH = 1

COMSOL_FOLDER = "./comsol/0.5x2cyl_E=1e5_nu=0.49/"
COMSOL_UNDEFORMED_FILENAME = COMSOL_FOLDER + "undeformed.stl"
COMSOL_DEFORMED_FILENAME = COMSOL_FOLDER + "deformed_output.txt"

SPACING = ROD_WIDTH * 2

class FigureType(Enum):
    MODEL = 0
    FEM = 1
    MODEL_FEM_COMPARISON = 2
    MODEL_XSECTION = 3
    FEM_XSECTION = 4
    MODEL_FEM_XSECTION_COMPARISON = 5

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODEL_FEM_COMPARISON]

def plotModel(undeformed_rod, deformed_rod, undeformed_rod_mesh, deformed_rod_mesh, undeformed_xsections, deformed_xsections):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")

    for p in deformed_rod_mesh.points:
        p += np.array([-SPACING,0,0])
    for xsection in deformed_xsections:
        for p in xsection.points:
            p += np.array([-SPACING,0,0])

    plotter.add_mesh(undeformed_rod_mesh, color=UNDEFORMED_COLOR, opacity=0.3)
    plotter.add_mesh(deformed_rod_mesh, color=DEFORMED_COLOR, opacity=0.3)

    for xsection in undeformed_xsections:
        plotter.add_mesh(xsection, color=UNDEFORMED_COLOR, opacity=0.7)
    for xsection in deformed_xsections:
        plotter.add_mesh(xsection, color=DEFORMED_COLOR, opacity=0.7)

    plotter.add_floor(pad=1)

    np.set_printoptions(precision=6)
    p_tip_undeformed = undeformed_rod.tipPosition()
    p_tip_deformed = deformed_rod.tipPosition()
    plotter.add_text(f"Undeformed tip position: {p_tip_undeformed}\nDeformed tip position:   {p_tip_deformed}", position='lower_left', font_size=14)

    plotter.show()


def plotFEM(undeformed_mesh, deformed_mesh):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")

    deformed_mesh.apply_translation([-SPACING,0,0])

    cross_section, undeformed_cross_section, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH/2)

    plotter.add_mesh(undeformed_mesh, color=[255,0,0], opacity=0.5)
    plotter.add_mesh(deformed_mesh, color=[0,0,255], opacity=0.5)
    plotter.add_mesh(undeformed_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_floor(pad=1)

    tip_cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH - 1e-5)

    plotter.add_text(f"Undeformed tip position: {tip_cross_section.centroid(undeformed_mesh.vertices)}\nDeformed tip position:   {tip_cross_section.centroid(deformed_mesh.vertices)}", position='lower_left', font_size=14)

    plotter.show()


def plotModelFEMComparison(undeformed_rod_mesh, deformed_rod_mesh, undeformed_xsections, deformed_xsections, undeformed_mesh, deformed_mesh):
    plotter = pv.Plotter()
    plotter.add_text("Model & FEM Results")
    plotter.camera_position = 'xz'

    # Plot model result
    for p in undeformed_rod_mesh.points:
        p += np.array([SPACING,0,0])
    for xsection in undeformed_xsections:
        for p in xsection.points:
            p += np.array([SPACING,0,0])

    plotter.add_mesh(undeformed_rod_mesh, color=UNDEFORMED_COLOR, opacity=0.4, specular=1.0, show_edges=False, edge_color='black', line_width=1)
    plotter.add_mesh(deformed_rod_mesh, color=DEFORMED_COLOR, opacity=0.4, specular=1.0, show_edges=False, edge_color='black', line_width=1)

    plotter.add_mesh(undeformed_xsections[NUM_ROD_NODES//2], color=UNDEFORMED_COLOR, opacity=0.7, show_edges=False, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_xsections[NUM_ROD_NODES//2], color=DEFORMED_COLOR, opacity=0.7, show_edges=False, edge_color='black', line_width=3)

    
    # Plot deformed FEM result
    deformed_mesh.apply_translation([-SPACING,0,0])
    _, _, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH/2)

    plotter.add_mesh(deformed_mesh, color=[0,0,255], opacity=0.4, specular=1.0, show_edges=False, edge_color='black', line_width=1)
    plotter.add_mesh(deformed_cross_section, color=DEFORMED_COLOR, opacity=0.7, show_edges=False, edge_color='black', line_width=3)
    plotter.add_floor(pad=1)

    plotter.show()


def plotModelXSection(undeformed_xsections_2d, deformed_xsections_2d):
    plotter = pv.Plotter()
    plotter.camera_position = 'xy'
    plotter.camera.position = [0,0,ROD_WIDTH*5]

    middle_node_num = NUM_ROD_NODES//2+1
    # middle_node_num = 4
    # plotter.add_text(f"Analytical Cross Section View at Node {middle_node_num}")

    plotter.add_mesh(deformed_xsections_2d[middle_node_num], color=DEFORMED_COLOR, opacity=0.5)
    plotter.add_mesh(undeformed_xsections_2d[middle_node_num], color=UNDEFORMED_COLOR, opacity=0.5)

    plotter.show()


def plotFEMXSection(undeformed_mesh, deformed_mesh):
    plotter = pv.Plotter()

    plotter.camera_position = 'xy'
    plotter.camera.position = [0, 0, ROD_WIDTH*5]

    cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH/2)
    
    undeformed_2d_cross_section = cross_section.asPolyData2D(undeformed_mesh.vertices)
    deformed_2d_cross_section = cross_section.asPolyData2D(deformed_mesh.vertices)

    # for p in deformed_2d_cross_section.points:
    #     p += np.array([-2,0,0])

    plotter.add_mesh(undeformed_2d_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_2d_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)

    # calculate a,b,c from x and y axes
    # mesh_a = np.linalg.norm(def_x - def_origin) / np.linalg.norm(undef_x - undef_origin)
    # mesh_b = np.linalg.norm(def_y - def_origin) / np.linalg.norm(undef_y - undef_origin)
    # bottom_str = f"Undeformed area: {utils.area(undeformed_2d_cross_section.points):.4f}\nDeformed area:    {utils.area(deformed_2d_cross_section.points):.4f}\nDeformed (a,b,c): {mesh_a:.4f}, {mesh_b:.4f}, {0:.4f}"
    # plotter.add_text(bottom_str,
    #                  position=(5, 5), font_size=10)

    plotter.show()


def plotModelFEMXSectionComparison(undeformed_xsections_2d, deformed_xsections_2d, undeformed_mesh, deformed_mesh):
    plotter = pv.Plotter()
    plotter.camera_position = 'xy'
    plotter.camera.position = [0,0,ROD_WIDTH*5]

    middle_node_num = NUM_ROD_NODES//2+1
    # middle_node_num = 4
    # plotter.add_text(f"Analytical Cross Section View at Node {middle_node_num}")

    plotter.add_mesh(deformed_xsections_2d[middle_node_num], color=DEFORMED_COLOR, opacity=0.5)
    plotter.add_mesh(undeformed_xsections_2d[middle_node_num], color=UNDEFORMED_COLOR, opacity=0.5)

    cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, ROD_LENGTH/2)
    
    undeformed_2d_cross_section = cross_section.asPolyData2D(undeformed_mesh.vertices)
    deformed_2d_cross_section = cross_section.asPolyData2D(deformed_mesh.vertices)

    plotter.show()

def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    rod = cosserat.CosseratRod(NUM_ROD_NODES, ROD_LENGTH, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH/2, ROD_WIDTH/2), 1e5, 0.49)
    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(0.5, 0.5), 1e5, 0.49)
    undeformed_rod = copy.copy(rod)

    undeformed_rod_mesh = rod.asMesh()
    undeformed_xsections = rod.nodeCrossSectionPolyData()
    undeformed_xsections_2d = rod.nodeCrossSectionPolyData2D()
    # rod.solveOptimizationProblem([10000,0,0], [0,1])
    print("Solving optimization problem...")
    rod.solveOptimizationProblem([0, 0, 50000])
    # rod.solveOptimizationProblemTorsionalMoment(-5000)
    deformed_rod_mesh = rod.asMesh()
    deformed_xsections = rod.nodeCrossSectionPolyData()
    deformed_xsections_2d = rod.nodeCrossSectionPolyData2D()


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

    print("Plotting...")
    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODEL):
            process_list.append(Process(target=plotModel, args=(undeformed_rod, rod, undeformed_rod_mesh, deformed_rod_mesh, undeformed_xsections, deformed_xsections)))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, args=(undeformed_mesh, deformed_mesh)))
        elif (fig_type == FigureType.MODEL_FEM_COMPARISON):
            process_list.append(Process(target=plotModelFEMComparison, args=(undeformed_rod_mesh, deformed_rod_mesh, undeformed_xsections, deformed_xsections, undeformed_mesh, deformed_mesh)))
        elif (fig_type == FigureType.MODEL_XSECTION):
            process_list.append(Process(target=plotModelXSection, args=(undeformed_xsections_2d, deformed_xsections_2d)))
        elif (fig_type == FigureType.FEM_XSECTION):
            process_list.append(Process(target=plotFEMXSection, args=(undeformed_mesh, deformed_mesh)))
        elif (fig_type == FigureType.MODEL_FEM_XSECTION_COMPARISON):
            process_list.append(Process(target=plotModelFEMXSectionComparison, args=(undeformed_xsections_2d, deformed_xsections_2d, undeformed_mesh, deformed_mesh)))
        
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()