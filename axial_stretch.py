import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

from enum import Enum
from multiprocessing import Process

NUM_ROD_NODES = 21
UNDEFORMED_COLOR = [255, 0, 0]
DEFORMED_COLOR = [0, 0, 255]
MODEL_COLOR = [255, 130, 0]
FEM_COLOR = [38, 227, 0]
ROD_LENGTH = 2
ROD_WIDTH = 1

# NASTRAN_FOLDER = "nastran/0.5x2_cyl_E=1e5_nu=0.49/"
NASTRAN_FOLDER = "nastran/0.5x2_cyl_E=1e5_nu=0.3/"
NASTRAN_UNDEFORMED_STL_FILENAME = NASTRAN_FOLDER + "undeformed.stl"
NASTRAN_UNDEFORMED_CSV_FILENAME = NASTRAN_FOLDER + "undeformed.csv"
# NASTRAN_DEFORMED_CSV_FILENAMES = ["deformed_F=-15000.csv", "deformed_F=-5000.csv", "deformed_F=20000.csv", "deformed_F=50000.csv"]
NASTRAN_DEFORMED_CSV_FILENAMES = ["deformed_F=50000.csv"]

SPACING = ROD_WIDTH * 2

class FigureType(Enum):
    MODELS = 0
    FEM = 1
    CROSS_SECTIONS = 2

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODELS, FigureType.FEM, FigureType.CROSS_SECTIONS]

# Z_FORCES = [-50000, -25000, -10000, 0, 10000, 25000, 50000]
# Z_FORCES = [-15000, -5000, 0, 20000, 50000]
Z_FORCES = [0, 50000]
# Z_FORCES = [0, -15000]
# Z_FORCES = [-10000, 0, 10000]

def plotModels(deformed_rods, undeformed_index=0):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")
    plotter.camera.position = [0, 5*ROD_WIDTH*len(deformed_rods), 2*ROD_LENGTH]

    tip_centroids = []
    for i,deformed_rod in enumerate(deformed_rods):
        mesh = deformed_rod.asMesh()
        mesh_disp = np.array([-SPACING*(len(deformed_rods)-1)/2 + SPACING*i, 0, 0])
        for p in mesh.points:
            p += np.array([-SPACING*(len(deformed_rods)-1)/2 + SPACING*i, 0, 0])
        
        plotter.add_mesh(mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)

        edges = mesh.extract_feature_edges(
            boundary_edges=False, non_manifold_edges=False, feature_angle=30, manifold_edges=False
        )
        plotter.add_mesh(edges, color="k")

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData(scale_factor=1.0)
        for xsection in deformed_xsections[1:-1]:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k', line_width=2)

        tip_centroids.append(deformed_rod.tipPosition())

    print(f"Model tip centroids:\n{tip_centroids}")

    plotter.add_floor()
    plotter.show()


def plotFEM(deformed_fem_meshes, undeformed_index=0):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH*len(deformed_fem_meshes), 2*ROD_LENGTH]

    cross_section, _, _ = mesh.getCrossSectionsMesh(deformed_fem_meshes[undeformed_index], deformed_fem_meshes[0], ROD_LENGTH-1e-4)

    tip_centroids = []
    for i,fem_mesh in enumerate(deformed_fem_meshes):
        tip_centroids.append(cross_section.centroid(fem_mesh.vertices))
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        fem_mesh.apply_translation( np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0]) )
        plotter.add_mesh(fem_mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    print(f"FEM tip centroids:\n{tip_centroids}")

    plotter.add_floor()
    plotter.show()

def plotCrossSections(deformed_rods, undeformed_fem_mesh, deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera_position = 'xy'
    plotter.camera.enable_parallel_projection()
    # plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    node_num = 19
    s_xsection = node_num * (ROD_LENGTH / (NUM_ROD_NODES-1))

    num_meshes = len(deformed_rods)
    # plot each rod
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        rod_mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(num_meshes-1)/2 + SPACING*i, 0, 0])
        
        for p in rod_mesh.points:
            p += mesh_disp
        
        rod_xsection = deformed_rod.nodeCrossSectionPolyData2D()[node_num]
        for p in rod_xsection.points:
            p += mesh_disp
        
        plotter.add_mesh(rod_xsection, color=MODEL_COLOR, opacity=0.5, show_edges=True)

    for i,fem_mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        mesh_disp = np.array([-SPACING*(num_meshes-1)/2 + SPACING*i, 0, 0])
        fem_mesh.apply_translation( mesh_disp )

        cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_fem_mesh, fem_mesh, s_xsection)
        deformed_2d_cross_section = cross_section.asPolyData2D(fem_mesh.vertices)
        plotter.add_mesh(deformed_2d_cross_section, color=FEM_COLOR, opacity=0.5, show_edges=True)

    plotter.show()

def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    rod = cosserat.CosseratRod(NUM_ROD_NODES, ROD_LENGTH, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH/2, ROD_WIDTH/2), 1e5, 0.3)
    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(0.5, 0.5), 1e5, 0.49)
    undeformed_rod = copy.copy(rod)

    deformed_rods = []
    for z_force in Z_FORCES:
        deformed_rod = copy.copy(rod)
        deformed_rod.solveOptimizationProblem([cosserat.AppliedTipForce([0,0,z_force], [0,0], True)])
        deformed_rods.append(deformed_rod)

    # plotModels(deformed_rods)


    ######################################################
    # Load FEM Results
    ######################################################
    print("Loading FEM results...")

    undeformed_fem_mesh = tm.load_mesh(NASTRAN_UNDEFORMED_STL_FILENAME)

    ## HOW TO GENERATE THIS OUTPUT FILE IN COMSOL (because I couldn't figure out how to export the deformed mesh):
    # 1. Run the FEM simulation
    # 2. Under Results, Right click 'Export', then click 'Data'
    # 3. Under 'Dataset', Select the solution
    # 4. Under 'Expressions', add 3 expressions: x+u, y+v, z+w (u, v, w are the x,y,z displacements)
    # 5. Under 'Output', change 'Geometry Level' to 'Surface' (i.e. only print data for surface nodes)
    # 6. Choose a filename and click 'Export' at the top
    deformed_fem_meshes = []
    for deformed_csv in NASTRAN_DEFORMED_CSV_FILENAMES:
        full_path = NASTRAN_FOLDER + deformed_csv
        deformed_fem_mesh = utils.getDeformedMeshFromNastranData(undeformed_fem_mesh, NASTRAN_UNDEFORMED_CSV_FILENAME, full_path)
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
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": np.array(deformed_rods), "undeformed_index": undeformed_index.item()} ))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, kwargs={"deformed_fem_meshes": np.array(deformed_fem_meshes), "undeformed_index": undeformed_index.item()} ))
        elif (fig_type == FigureType.CROSS_SECTIONS):
            process_list.append(Process(target=plotCrossSections, kwargs={"deformed_rods": np.array(deformed_rods), "undeformed_fem_mesh": undeformed_fem_mesh, "deformed_fem_meshes": deformed_fem_meshes}))
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()