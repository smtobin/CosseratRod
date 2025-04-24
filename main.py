import cosserat
import mesh
import utils

import pyvista as pv
import numpy as np
import trimesh as tm
import copy

NUM_ROD_NODES = 11
UNDEFORMED_COLOR = [255, 0, 0]
DEFORMED_COLOR = [0, 0, 255]

COMSOL_FOLDER = "./comsol/0.5x2cyl_E=1e5_nu=0.49/"
COMSOL_UNDEFORMED_FILENAME = COMSOL_FOLDER + "undeformed.stl"
COMSOL_DEFORMED_FILENAME = COMSOL_FOLDER + "deformed_output.txt"

def main():

    # rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalEllipseCrossSection(0.5, 0.5), 1e5, 0.49)
    rod = cosserat.CosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(0.5, 0.5), 1e5, 0.49)
    l_rod = cosserat.LinearDeformationCosseratRod(NUM_ROD_NODES, 2, cosserat.AnalyticalRectCrossSection(0.5, 0.5), 1e5, 0.49)

    undeformed_rod = copy.copy(rod)

    undeformed_rod_mesh = rod.asMesh()
    undeformed_xsections = rod.nodeCrossSectionPolyData()
    undeformed_xsections_2d = rod.nodeCrossSectionPolyData2D()
    # rod.solveOptimizationProblem([10000,0,0], [0,1])
    applied_force = [500,0,0]
    rod.solveOptimizationProblem(applied_force)
    l_rod.solveOptimizationProblem(applied_force)
    # rod.solveOptimizationProblemTorsionalMoment(-5000)
    deformed_rod_mesh = rod.asMesh()
    deformed_xsections = rod.nodeCrossSectionPolyData()
    deformed_xsections_2d = rod.nodeCrossSectionPolyData2D()

    deformed_l_rod_mesh = l_rod.asMesh()
    deformed_l_rod_xsections = l_rod.nodeCrossSectionPolyData()
    deformed_l_rod_xsections_2d = l_rod.nodeCrossSectionPolyData2D()



    plotter = pv.Plotter(shape=(2,2))



    ###############################################
    # plot analytical deformed and undeformed rods
    ###############################################
    plotter.subplot(0,0)
    plotter.add_text("Model Results")

    for p in deformed_rod_mesh.points:
        p += np.array([2,0,0])
    for xsection in deformed_xsections:
        for p in xsection.points:
            p += np.array([2,0,0])

    for p in deformed_l_rod_mesh.points:
        p += np.array([4,0,0])
    for xsection in deformed_l_rod_xsections:
        for p in xsection.points:
            p += np.array([4,0,0])

    plotter.add_mesh(undeformed_rod_mesh, color=UNDEFORMED_COLOR, opacity=0.3)
    plotter.add_mesh(deformed_rod_mesh, color=DEFORMED_COLOR, opacity=0.3)
    plotter.add_mesh(deformed_l_rod_mesh, color=DEFORMED_COLOR, opacity=0.3 )

    for xsection in undeformed_xsections:
        plotter.add_mesh(xsection, color=UNDEFORMED_COLOR, opacity=0.7)
    for xsection in deformed_xsections:
        plotter.add_mesh(xsection, color=DEFORMED_COLOR, opacity=0.7)
    for xsection in deformed_l_rod_xsections:
        plotter.add_mesh(xsection, color=DEFORMED_COLOR, opacity=0.7)

    plotter.add_floor(pad=1)

    np.set_printoptions(precision=6)
    p_tip_undeformed = undeformed_rod.tipPosition()
    p_tip_deformed = rod.tipPosition()
    plotter.add_text(f"Undeformed tip position: {p_tip_undeformed}\nDeformed tip position:   {p_tip_deformed}", position='lower_left', font_size=14)



    ##########################################################
    # plot analytical deformed and undeformed cross sections
    ##########################################################
    plotter.subplot(1,0)
    plotter.camera_position = 'xy'
    plotter.camera.position = [0,0,10]

    middle_node_num = NUM_ROD_NODES//2
    middle_node_num = 2
    # middle_node_num = 4
    plotter.add_text(f"Analytical Cross Section View at Node {middle_node_num}")

    plotter.add_mesh(deformed_xsections_2d[middle_node_num], color=DEFORMED_COLOR, opacity=0.5)
    plotter.add_mesh(undeformed_xsections_2d[middle_node_num], color=UNDEFORMED_COLOR, opacity=0.5)
    plotter.add_mesh(deformed_l_rod_xsections_2d[middle_node_num], color=DEFORMED_COLOR, opacity=0.5)


    # bottom_str = f"Undeformed area: {undeformed_rod.cross_section.A0:.4f}\nDeformed area:    {rod.cross_section.deformedArea(rod.nodeDistortionMatrix(middle_node_num)):.4f}\nDeformed (a,b,c): {rod.nodeDistortionMatrix(middle_node_num)[0,0]:.4f}, {rod.nodeDistortionMatrix(middle_node_num)[1,1]:.4f}, {rod.nodeDistortionMatrix(middle_node_num)[0,1]:.4f}"
    # plotter.add_text(bottom_str,
    #                  position=(5, 5), font_size=10)
    # plotter.add_text(f"Deformed (a,b,c): {rod.nodeDistortionMatrix(middle_node_num)[0,0]:.4f}, {rod.nodeDistortionMatrix(middle_node_num)[1,1]:.4f}, {rod.nodeDistortionMatrix(middle_node_num)[0,1]:.4f}",
    #                  position=(5, 40), font_size=10)
    


    ############################################################
    # plot FEM deformed and undeformed meshes
    ############################################################
    plotter.subplot(0,1)

    plotter.add_text(f"FEM Results")

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

    deformed_mesh.apply_translation([2,0,0])

    undeformed_vertices = undeformed_mesh.vertices
    deformed_vertices = deformed_mesh.vertices

    cross_section, undeformed_cross_section, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, 1.0)
    undeformed_centroid = cross_section.centroid(undeformed_vertices)
    print(f"undeformed centroid: {undeformed_centroid}")

    deformed_centroid = cross_section.centroid(deformed_vertices)
    print(f"deformed centroid: {deformed_centroid}")

    [undef_origin, undef_x, undef_y] = cross_section.axes(undeformed_vertices)
    [def_origin, def_x, def_y] = cross_section.axes(deformed_vertices)

    plotter.add_mesh(undeformed_mesh, color=[255,0,0], opacity=0.5)
    plotter.add_mesh(deformed_mesh, color=[0,0,255], opacity=0.5)
    plotter.add_mesh(undeformed_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_floor(pad=1)

    tip_cross_section, tip_undeformed_cross_section, tip_deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, 1.999)

    plotter.add_text(f"Undeformed tip position: {tip_cross_section.centroid(undeformed_vertices)}\nDeformed tip position:   {tip_cross_section.centroid(deformed_vertices)}", position='lower_left', font_size=14)

    # plotter.add_lines(np.array([undef_origin, undef_x]))
    # plotter.add_lines(np.array([undef_origin, undef_y]))
    # plotter.add_lines(np.array([def_origin, def_x]))
    # plotter.add_lines(np.array([def_origin, def_y]))

    ##############################################################
    # plot FEM deformed and undeformed cross sections
    ##############################################################
    plotter.subplot(1,1)
    plotter.add_text(f"FEM Cross Section View")

    plotter.camera_position = 'xy'
    plotter.camera.position = [0, 0, 10]

    

    # cross_section, undeformed_cross_section, deformed_cross_section = mesh.getCrossSectionsMesh(undeformed_mesh, deformed_mesh, 2.)
    undeformed_2d_cross_section = cross_section.asPolyData2D(undeformed_vertices)
    deformed_2d_cross_section = cross_section.asPolyData2D(deformed_vertices)

    for p in deformed_2d_cross_section.points:
        p += np.array([-2,0,0])

    plotter.add_mesh(undeformed_2d_cross_section, color='red', opacity=0.7, show_edges=True, edge_color='black', line_width=3)
    plotter.add_mesh(deformed_2d_cross_section, color='blue', opacity=0.7, show_edges=True, edge_color='black', line_width=3)

    # calculate a,b,c from x and y axes
    mesh_a = np.linalg.norm(def_x - def_origin) / np.linalg.norm(undef_x - undef_origin)
    mesh_b = np.linalg.norm(def_y - def_origin) / np.linalg.norm(undef_y - undef_origin)
    bottom_str = f"Undeformed area: {utils.area(undeformed_2d_cross_section.points):.4f}\nDeformed area:    {utils.area(deformed_2d_cross_section.points):.4f}\nDeformed (a,b,c): {mesh_a:.4f}, {mesh_b:.4f}, {0:.4f}"
    plotter.add_text(bottom_str,
                     position=(5, 5), font_size=10)

    plotter.show()

if __name__ == '__main__':
    main()