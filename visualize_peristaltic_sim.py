
import cosserat
import mesh
import utils
import vtk

import os
import time

import pyvista as pv
import numpy as np

ROD_FILENAME = "cxx/output/peristaltic.txt"
SIM_FOLDER_PATH = "cxx/output/sim"
# SIM_FOLDER_PATH = "cxx/output/N=13_circle_sim_2actuator"

MODEL_COLOR = [255, 130, 0]

def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True

def readSimFromFolder(folder_path):
    filenames = sorted(os.listdir(folder_path))
    robot_filename = os.path.join(folder_path, filenames[0]) # should be robot.txt
    with open(robot_filename, 'r') as file:
        data = file.read()
        data_arr = data.split("\n")
    
    N = int(data_arr[0])
    length = float(data_arr[1])
    E = float(data_arr[2])
    nu = float(data_arr[3])
    cs_type = data_arr[4]
    cs_rx = float(data_arr[5])
    cs_ry = float(data_arr[6])

    if cs_type == "Ellipse":
        cross_section = cosserat.AnalyticalEllipseCrossSection(cs_rx, cs_ry)
    elif cs_type == "Rect":
        cross_section = cosserat.AnalyticalRectCrossSection(cs_rx*2, cs_ry*2)

    rod = cosserat.CosseratRod(N, length, cross_section, E, nu)

    actuator_pressures = []
    positions = []
    orientations = []
    states = []
    for step_filename in filenames[1:]:
        with open(os.path.join(folder_path, step_filename)) as file:

            # extract actuator pressures
            pressures = [int(s) for s in file.readline().split(" ") if represents_int(s)]
            actuator_pressures.append(pressures)

            # extract rod state
            data = file.read()
            data_arr = data.split("\n")

            expected_states = 9*N-6
            state = np.array(data_arr[:expected_states]).astype(float)
            position = np.array(data_arr[expected_states:expected_states+3]).astype(float)
            if len(data_arr) > expected_states+3:
                orientation = np.array(data_arr[expected_states+3:expected_states+6]).astype(float)
            else:
                orientation = np.array([0,0,0])
            states.append(state)
            positions.append(position)
            orientations.append(utils.MatExp_so3(orientation/np.linalg.norm(orientation), np.linalg.norm(orientation)))

    return rod, actuator_pressures, positions, orientations, states
       
    

def main():
    

    (rod, actuator_pressures, positions, orientations, states) = readSimFromFolder(SIM_FOLDER_PATH)

    rod_mesh = rod.asMesh(rod.n//2, positions[0], orientations[0])


    ##########################################
    # Set up plotter
    ###########################################
    pl = pv.Plotter()
    # pl.camera.enable_parallel_projection()
    # pl.camera_position = 'xy'
    pl.camera_position = [5,0,25]
    # pl.camera.zoom(0.3)
    # pl.camera.position = [0, 0, 1]
    pl.camera.focal_point = [4.99, 0, 0]
    # pl.camera.clipping_range = (0.01, 1000.01)

    plane = pv.Plane()
    pl.add_mesh(plane)

    ############################################
    # Create initial meshes (that will be updated each time step)
    ############################################

    # the rod
    pv_mesh = pv.PolyData(rod_mesh.points, faces=rod_mesh.faces)
    actor = pl.add_mesh(pv_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0)

    # one for each cross section
    xsections = rod.nodeCrossSectionPolyData(rod.n//2, positions[0], orientations[0])
    cross_section_meshes = []
    for i,xsection in enumerate(xsections):
        pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
        cross_section_meshes.append(pv_cs_mesh)
        pl.add_mesh(cross_section_meshes[i], color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
    
    # pl.add_floor()

    axes = pl.add_axes_at_origin()

    r1 = 4  
    c1 = np.array([0,r1])
    circle1 = pv.Circle(r1)
    for pt in circle1.points:
        pt[0] = pt[0] + c1[0]
        pt[1] = pt[1] + c1[1]
    pl.add_mesh(circle1)

    # Real cos45 = 0.5*std::sqrt(2.0);
    #     Real sin45 = 0.5*std::sqrt(2.0);
    #     Real r1 = 4.0; Real r2 = 0.5; Real r3 = 1.0;
    #     Vec2r c1(0, r1);
    #     Vec2r c2 = c1 + r1*Vec2r(cos45, -sin45) + r2*Vec2r(-cos45, sin45);
    #     Vec2r c3 = c2 + r2*Vec2r(-cos45, sin45) + r3*Vec2r(-cos45, sin45);
    r2 = 3.0
    c2 = c1 + r1*np.array([np.sqrt(2)/2, -np.sqrt(2)/2]) + r2*np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
    circle2 = pv.Circle(r2)
    for pt in circle2.points:
        pt[0] = pt[0] + c2[0]
        pt[1] = pt[1] + c2[1]
    pl.add_mesh(circle2, color=[0,0,0], style='wireframe')

    r3 = 2.5
    c3 = c2 + (r2+r3)*np.array([1, 0])
    circle3 = pv.Circle(r3)
    for pt in circle3.points:
        pt[0] = pt[0] + c3[0]
        pt[1] = pt[1] + c3[1]
    pl.add_mesh(circle3, color=[0,255,0])
    

    # start plotting
    pl.show(auto_close=False, interactive_update=True)


    #########################################
    # Animate through the sim states
    #########################################

    # loop through the simulation states and update the meshes
    for step in range(len(states)):
        # update the rod state and get its new mesh
        rod.Z = states[step]
        new_rod_mesh = rod.asMesh(rod.n//2, positions[step], orientations[step])

        new_pv_mesh = pv.PolyData(new_rod_mesh.points, faces=new_rod_mesh.faces)
        pv_mesh.shallow_copy(new_pv_mesh)

        # update the cross section meshes
        new_xsections = rod.nodeCrossSectionPolyData(rod.n//2, positions[step], orientations[step])
        for i,xsection in enumerate(new_xsections):
            new_pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
            cross_section_meshes[i].shallow_copy(new_pv_cs_mesh)

        pl.render()
        time.sleep(1/30)

    time.sleep(10)
    pl.close()

if __name__ == "__main__":
    main()