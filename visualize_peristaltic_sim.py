
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
# SIM_FOLDER_PATH = "cxx/output/N=49_bent_sim"

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
    pl.camera.position = [0, 0.1, 5]
    pl.camera.focal_point = [0, 0, 0]
    pl.camera.clipping_range = (0.01, 1000.01)

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
    
    pl.add_floor()
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
        time.sleep(1/10)

    time.sleep(10)
    pl.close()

if __name__ == "__main__":
    main()