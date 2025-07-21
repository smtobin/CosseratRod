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
SPACING = ROD_WIDTH_X * 2

NASTRAN_UNDEFORMED_STL_FILENAME = "nastran/1x0.5_block_E=1e5_nu=0.3/undeformed.stl"
NASTRAN_UNDEFORMED_CSV_FILENAME = "nastran/1x0.5_block_E=1e5_nu=0.3/undeformed.csv"
NASTRAN_DEFORMED_FILENAME = "nastran/1x0.5_block_E=1e5_nu=0.3/deformed_F=500.csv"
SOLVED_ROD_FILENAMES = ["cxx/output/RodCSDLin_N=15_F=(0,500,0).txt"]

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
        
        plotter.add_mesh(mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)
        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
        for xsection in deformed_xsections:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
       

    plotter.add_floor()
    plotter.show()

def plotModelFEM(deformed_rods, undeformed_fem_mesh, deformed_fem_meshes):
    plotter = pv.Plotter()
    plotter.add_text("FEM Results")
    plotter.camera.position = [0, 5*ROD_WIDTH_X*len(deformed_fem_meshes), ROD_LENGTH]

    num_meshes = len(deformed_rods) + len(deformed_fem_meshes)
    fem_tip_centroids = []
    for i,fem_mesh in enumerate(deformed_fem_meshes):
        cross_section, _, _ = mesh.getCrossSectionsMesh(undeformed_fem_mesh, fem_mesh, ROD_LENGTH-1e-4)
        fem_tip_centroids.append(cross_section.centroid(fem_mesh.vertices))
        
        fem_mesh.apply_translation( np.array([-SPACING*(num_meshes-1)/2 + SPACING*i, 0, 0]) )
        plotter.add_mesh(fem_mesh, color=FEM_COLOR, opacity=1, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=True)
        fem_tip_centroids.append
    # plot each rod
    model_tip_centroids = []
    for i,deformed_rod in enumerate(deformed_rods):
        # get mesh from Cosserat rod class
        rod_mesh = deformed_rod.asMesh()

        # move mesh along x-axis to be separate from other meshes
        mesh_disp = np.array([-SPACING*(num_meshes-1)/2 + SPACING*(len(deformed_fem_meshes) + i), 0, 0])
        
        for p in rod_mesh.points:
            p += mesh_disp
        
        plotter.add_mesh(rod_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)

        edges = rod_mesh.extract_feature_edges(
            boundary_edges=False, non_manifold_edges=False, feature_angle=30, manifold_edges=False
        )
        plotter.add_mesh(edges, color="k")

        # plot cross sections
        deformed_xsections = deformed_rod.nodeCrossSectionPolyData(scale_factor=1.0)
        for xsection in deformed_xsections[1:-1]:
            for p in xsection.points:
                p += mesh_disp

            plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k', line_width=2)
        model_tip_centroids.append(deformed_rod.tipPosition())

    print(f"Fem centroids: {fem_tip_centroids}")
    print(f"Model centroids: {model_tip_centroids}")

    plotter.add_floor()
    plotter.show()

def main():
    undeformed_fem_mesh = tm.load_mesh(NASTRAN_UNDEFORMED_STL_FILENAME)
    deformed_fem_mesh = utils.getDeformedMeshFromNastranData(undeformed_fem_mesh, NASTRAN_UNDEFORMED_CSV_FILENAME, NASTRAN_DEFORMED_FILENAME)

    undeformed_rods = []
    deformed_rods = []
    for filename in SOLVED_ROD_FILENAMES:
        undeformed_rod, deformed_rod = utils.loadRodFromFile(filename)
        print(f"Total energy: {deformed_rod._totalEnergy(deformed_rod.Z, [cosserat.AppliedTipForce([0,500,0], [0,0], True)])}")
        
        mesh.meshRodVertexError(undeformed_fem_mesh, deformed_fem_mesh, undeformed_rod, deformed_rod)
        
        undeformed_rods.append(undeformed_rod)
        deformed_rods.append(deformed_rod)
    
    plotModels(deformed_rods)

if __name__ == "__main__":
    main()