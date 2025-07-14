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
TENDON_COLOR = [62, 62, 62]
UNDEFORMED_COLOR = [255, 130, 0]
ARROW_COLOR = [255, 0, 0]
FEM_COLOR = [38, 227, 0]
ROD_LENGTH = 4
ROD_WIDTH_X = 1
ROD_WIDTH_Y = 1

SPACING = ROD_WIDTH_X * 2

class FigureType(Enum):
    MODELS = 0
    CROSS_SECTIONS = 1

# set the types of figures to create
FIGURE_TYPES = [FigureType.MODELS]

Y_FORCES = [5000]
AB_COORDS = [[0,0]]

def plotModels(deformed_rods, undeformed_index=0):
    plotter = pv.Plotter()
    plotter.add_text("Model Results")
    plotter.camera.position = [0, -5*ROD_WIDTH_X*len(deformed_rods), ROD_LENGTH]

    # plot each rod
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        rod_mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(len(deformed_rods)-1)/2 + SPACING*i, 0, 0])
        
        for p in rod_mesh.points:
            p += mesh_disp

        if i == undeformed_index:
            mesh_color = UNDEFORMED_COLOR
        else:
            mesh_color = MODEL_COLOR
        
        plotter.add_mesh(rod_mesh, color=mesh_color, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)

        edges = rod_mesh.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False, feature_angle=30, manifold_edges=False
        )
        plotter.add_mesh(edges, color="k")

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
        for xsection in deformed_xsections:
            for p in xsection.points:
                p += mesh_disp
            
            
            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=False, edge_color='k')

            edges = xsection.extract_feature_edges(
                boundary_edges=True, non_manifold_edges=False, feature_angle=30, manifold_edges=False
            )
            plotter.add_mesh(edges, color="k", line_width=2)
       

        for tendon in deformed_rod.tendons:
            tendon_mesh = tendon.asMesh(deformed_rod)
            for p in tendon_mesh.points:
                p += mesh_disp
            
            plotter.add_mesh(tendon_mesh, color=TENDON_COLOR, opacity=1.0, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)

    plotter.add_floor()
    plotter.show()


def plotCrossSections(deformed_rods):
    plotter = pv.Plotter()
    plotter.add_text("Cross Sections")
    plotter.camera_position = 'xy'
    plotter.camera.enable_parallel_projection()
    # plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    node_num = 2
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
        
        
        plotter.add_mesh(rod_xsection, color=MODEL_COLOR, opacity=1.0, show_edges=False)

        edges = rod_xsection.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False, feature_angle=30, manifold_edges=False
        )
        plotter.add_mesh(edges, color="k", line_width=2)

    plotter.show()

def main():

    ########################################################
    # Compute analytical model results
    ########################################################
    tendons = [
        cosserat.Tendon(200000+100000, [ROD_WIDTH_X/2, 0, ROD_LENGTH/2]),
        cosserat.Tendon(100000, [0, ROD_WIDTH_Y/2, ROD_LENGTH/2]),
        cosserat.Tendon(100000, [-ROD_WIDTH_X/2, 0, ROD_LENGTH/2]),
        cosserat.Tendon(100000, [0, -ROD_WIDTH_Y/2, ROD_LENGTH/2]),
        cosserat.Tendon(100000, [np.sqrt(2)*ROD_WIDTH_X/4, np.sqrt(2)*ROD_WIDTH_Y/4, ROD_LENGTH]),
        cosserat.Tendon(50000+100000, [-np.sqrt(2)*ROD_WIDTH_X/4, np.sqrt(2)*ROD_WIDTH_Y/4, ROD_LENGTH]),
        cosserat.Tendon(50000+100000, [-np.sqrt(2)*ROD_WIDTH_X/4, -np.sqrt(2)*ROD_WIDTH_Y/4, ROD_LENGTH]),
        cosserat.Tendon(0+100000, [np.sqrt(2)*ROD_WIDTH_X/4, -np.sqrt(2)*ROD_WIDTH_Y/4, ROD_LENGTH])
    ]
    rod = cosserat.TendonActuatedCosseratRod(NUM_ROD_NODES, ROD_LENGTH, 
                                            cosserat.AnalyticalEllipseCrossSection(ROD_WIDTH_X/2, ROD_WIDTH_Y/2),
                                            tendons,
                                            3e6, 0.49)
    undeformed_rod = copy.copy(rod)

    deformed_rods = []
    # deformed_rods.append(undeformed_rod)
    for y_force, ab_coords in zip(Y_FORCES, AB_COORDS):

        # deformed_l_rod = copy.copy(l_rod)
        # deformed_l_rod.solveOptimizationProblem([0,y_force,0], ab_coords)
        # deformed_rods.append(deformed_l_rod)

        deformed_rod = copy.copy(rod)
        applied_forces = [
            # cosserat.AppliedTipForce([0,0,100000], [0,0], True)
        ]
        deformed_rod.solveOptimizationProblem(applied_forces)
        deformed_rods.append(deformed_rod)
    

    # spawn separate processes, one for each plot
    process_list = []
    for fig_type in FIGURE_TYPES:
        if (fig_type == FigureType.MODELS):
            process_list.append(Process(target=plotModels, kwargs={"deformed_rods": deformed_rods}))
        elif (fig_type == FigureType.CROSS_SECTIONS):
            process_list.append(Process(target=plotCrossSections, kwargs={"deformed_rods": deformed_rods}))
        process_list[-1].start()

    for elem in process_list:
        elem.join()

if __name__ == '__main__':
    main()