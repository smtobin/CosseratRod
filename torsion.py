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
UNDEFORMED_COLOR = [255, 130, 0]
ARROW_COLOR = [255, 0, 0]
FEM_COLOR = [38, 227, 0]
ROD_LENGTH = 4
ROD_WIDTH_X = 1
ROD_WIDTH_Y = 1

NASTRAN_FOLDER = "nastran/1x1x4_block_E=1e5_nu=0.49/"
NASTRAN_UNDEFORMED_STL_FILENAME = NASTRAN_FOLDER + "undeformed.stl"
# NASTRAN_UNDEFORMED_CSV_FILENAME = NASTRAN_FOLDER + "undeformed.csv"
# NASTRAN_DEFORMED_CSV_FILENAMES = ["deformed_M=100.stl"]
NASTRAN_DEFORMED_STL_FILENAMES = ["deformed_M=3000.stl"]

SPACING = ROD_WIDTH_X * 2

class FigureType(Enum):
    MODELS = 0
    FEM = 1
    MODELS_AND_FEM = 2
    CROSS_SECTIONS = 3

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODELS_AND_FEM]

# Y_FORCES = [0, 7500, 7500, 7500]
# AB_COORDS = [[0,0], [0,0], [0.5,0], [1,0]]

# Y_FORCES = [500, 500, 500]
# AB_COORDS = [[0,0], [0.5, 0.5], [1.0, 1.0]]

Z_MOMENTS = [3000]
# XY_COORDS = [[ROD_WIDTH_X/2,0]]

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

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
       

    plotter.add_floor()
    plotter.show()


def plotFEM(deformed_fem_meshes, undeformed_index=0):
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


def plotModelFEM(deformed_rods, undeformed_fem_mesh, deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    num_meshes = len(deformed_rods) + len(deformed_fem_meshes)
    for i,fem_mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        cross_section, undeformed_cross_section, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_fem_mesh, fem_mesh, ROD_LENGTH-1e-4)
        [undef_origin, undef_x, undef_y] = cross_section.axes(undeformed_fem_mesh.vertices)
        [def_origin, def_x, def_y] = cross_section.axes(fem_mesh.vertices)
        undef_x_axis = undef_x - undef_origin
        def_x_axis = def_x - def_origin
        theta = np.arccos(np.dot(undef_x_axis, def_x_axis) / (np.linalg.norm(undef_x_axis) * np.linalg.norm(def_x_axis)) )
        print(f"FEM angle change: {theta}")
        fem_mesh.apply_translation( np.array([-SPACING*(num_meshes-1)/2 + SPACING*i, 0, 0]) )
        plotter.add_mesh(fem_mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)

    # plot each rod
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        rod_mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(num_meshes-1)/2 + SPACING*(len(deformed_fem_meshes) + i), 0, 0])
        
        for p in rod_mesh.points:
            p += mesh_disp
        
        plotter.add_mesh(rod_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, feature_angle=50, show_edges=False)

        edges = rod_mesh.extract_feature_edges(
            boundary_edges=False, non_manifold_edges=False, feature_angle=50, manifold_edges=False
        )
        plotter.add_mesh(edges, color="k")

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData(scale_factor=1.0)
        for xsection in deformed_xsections[:-1]:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k', line_width=2)

        T_tip = deformed_rod.nodeTransforms()[-1]
        def_x_axis = T_tip[0:3,0]
        undef_x_axis = np.array([1,0,0])
        theta = np.arccos(np.dot(undef_x_axis, def_x_axis) / (np.linalg.norm(undef_x_axis) * np.linalg.norm(def_x_axis)) )
        print(f"Model angle change: {theta}")

    plotter.add_floor()
    plotter.show()

def plotCrossSections(deformed_rods, undeformed_fem_mesh, deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera_position = 'xy'
    plotter.camera.enable_parallel_projection()
    # plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    node_num = 2
    s_xsection = node_num * (ROD_LENGTH / (NUM_ROD_NODES-1))

    num_meshes = len(deformed_rods) + len(deformed_fem_meshes)
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
        
        plotter.add_mesh(rod_xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True)

    for i,fem_mesh in enumerate(deformed_fem_meshes):
    #     for p in mesh.points:
    #         p += np.array([-SPACING*(len(deformed_fem_meshes)-1)/2 + SPACING*i, 0, 0])
        fem_mesh.apply_translation( np.array([-SPACING*(num_meshes-1)/2 + SPACING*(len(deformed_rods) + i), 0, 0]) )

        cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_fem_mesh, fem_mesh, s_xsection)
        deformed_2d_cross_section = cross_section.asPolyData2D(fem_mesh.vertices)
        plotter.add_mesh(deformed_2d_cross_section, color=FEM_COLOR, opacity=1, show_edges=True)

    plotter.show()

def main():

    ########################################################
    # Compute analytical model results
    ########################################################

    torsional_correction = 0.141 / (1/6) # for a square cross section

    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
    if FigureType.MODELS in FIGURE_TYPES or FigureType.MODELS_AND_FEM in FIGURE_TYPES:
        rod = cosserat.CosseratRod(NUM_ROD_NODES, ROD_LENGTH, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
        # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH_X/2, ROD_WIDTH_X/2), 1e5, 0.49)
        l_rod = cosserat.LinearDeformationCosseratRod(NUM_ROD_NODES, ROD_LENGTH, cosserat.AnalyticalRectCrossSection(ROD_WIDTH_X, ROD_WIDTH_Y), 1e5, 0.49)
        undeformed_rod = copy.copy(rod)

        deformed_rods = []
        # deformed_rods.append(undeformed_rod)
        for z_moment in Z_MOMENTS:
            # deformed_l_rod = copy.copy(l_rod)
            # applied_forces = [cosserat.AppliedTipForce([0,y_force,0], [ROD_WIDTH_X/2,0], False),
            #                   cosserat.AppliedTipForce([0,-y_force,0], [-ROD_WIDTH_X/2,0], False)]
            # deformed_l_rod.solveOptimizationProblem(applied_forces)
            # deformed_rods.append(deformed_l_rod)

            force = (z_moment/2) / (ROD_WIDTH_X/2)

            applied_forces = []
            applied_moments = [cosserat.AppliedTipMoment([0,0,z_moment])]

            deformed_rod2 = copy.copy(rod)
            deformed_rod2.solveOptimizationProblemWithMomentsAndTorsionalCorrection(applied_forces, applied_moments, torsional_correction)
            deformed_rods.append(deformed_rod2)

            deformed_rod = copy.copy(rod)
            # applied_forces = [cosserat.AppliedTipForce([0,force,0], [ROD_WIDTH_X/2,ROD_WIDTH_Y/2], True),
            #                   cosserat.AppliedTipForce([0,-force,0], [-ROD_WIDTH_X/2,-ROD_WIDTH_Y/2], True)]
            deformed_rod.solveOptimizationProblemWithMoments(applied_forces, applied_moments)
            deformed_rods.append(deformed_rod)

            
            
            

    ######################################################
    # Load FEM Results
    ######################################################
    print("Loading FEM results...")

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
    undeformed_fem_mesh = tm.load_mesh(NASTRAN_UNDEFORMED_STL_FILENAME)

    deformed_fem_meshes = []
    for deformed_csv in NASTRAN_DEFORMED_STL_FILENAMES:
        full_path = NASTRAN_FOLDER + deformed_csv
        # deformed_fem_mesh = utils.getDeformedMeshFromNastranData(undeformed_fem_mesh, NASTRAN_UNDEFORMED_CSV_FILENAME, full_path)
        deformed_fem_mesh = tm.load_mesh(full_path)
        deformed_fem_meshes.append(deformed_fem_mesh)
    

    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODELS):
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": deformed_rods}))
        elif (fig_type == FigureType.FEM):
            process_list.append(Process(target=plotFEM, kwargs={"deformed_fem_meshes": deformed_fem_meshes}))
        elif (fig_type == FigureType.MODELS_AND_FEM):
            process_list.append(Process(target=plotModelFEM, kwargs={"deformed_rods": deformed_rods, "undeformed_fem_mesh": undeformed_fem_mesh, "deformed_fem_meshes": deformed_fem_meshes}))
        elif (fig_type == FigureType.CROSS_SECTIONS):
            process_list.append(Process(target=plotCrossSections, kwargs={"deformed_rods": deformed_rods, "undeformed_fem_mesh": undeformed_fem_mesh, "deformed_fem_meshes": deformed_fem_meshes}))
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()