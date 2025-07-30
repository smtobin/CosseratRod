
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
    states = []
    for step_filename in filenames[1:]:
        with open(os.path.join(folder_path, step_filename)) as file:

            # extract actuator pressures
            pressures = [int(s) for s in file.readline().split(" ") if represents_int(s)]
            actuator_pressures.append(pressures)

            # extract rod state
            data = file.read()
            data_arr = data.split("\n")
            state = np.array(data_arr[:-3]).astype(float)
            position = np.array(data_arr[-3:]).astype(float)
            states.append(state)
            positions.append(position)

    return rod, actuator_pressures, positions, states

def plotInitialModel(plotter, deformed_rod):

    # get mesh from Cosserat rod class
    rod_mesh = deformed_rod.asMesh()

    rod_actor = plotter.add_mesh(rod_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0, smooth_shading=True, split_sharp_edges=True, show_edges=False)
    # plot cross sections
    deformed_xsections = deformed_rod.nodeCrossSectionPolyData()
    cross_section_actors = []
    cross_section_meshes = []
    for xsection in deformed_xsections:
        cross_section_meshes.append(xsection)
        cs_actor = plotter.add_mesh(xsection, color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')
        cross_section_actors.append(cs_actor)

    plotter.show()

    new_rod_mesh = deformed_rod.asMesh()
    new_rod_mesh.points[:,1] = 1
    rod_mesh.shallow_copy(new_rod_mesh)
    plotter.render()

    return rod_mesh, cross_section_meshes, rod_actor, cross_section_actors
       

def update_callback(step):
    print(step)
    time.sleep(1)
    

def main():
    

    (rod, actuator_pressures, positions, states) = readSimFromFolder(SIM_FOLDER_PATH)
    rod_mesh = rod.asMesh(rod.n//2, positions[0])
    rod_mesh.points += positions[0]

    

    # plotter = pv.Plotter()
    # (rod_mesh, cross_section_meshes, rod_actor, cross_section_actors) = plotInitialModel(plotter, rod)

    # # plotter.show(auto_close=False, interactive_update=True)

    # for step in range(len(states)):
    #     rod.Z = states[step]
    #     new_mesh = rod.asMesh()
    #     rod_mesh.points = new_mesh.points
    #     plotter.update()
    #     time.sleep(1)

    # plotter.close()

    pl = pv.Plotter()
    pl.camera.position = [0, -10, 5]
    pv_mesh = pv.PolyData(rod_mesh.points, faces=rod_mesh.faces)
    actor = pl.add_mesh(pv_mesh, color=MODEL_COLOR, opacity=0.7, specular=1.0)
    pl.show(auto_close=False, interactive_update=True)

    for step in range(len(states)):
        rod.Z = states[step]
        new_rod_mesh = rod.asMesh(rod.n//2, positions[step])
        new_rod_mesh.points += positions[step]

        new_pv_mesh = pv.PolyData(new_rod_mesh.points, faces=new_rod_mesh.faces)
        pv_mesh.shallow_copy(new_pv_mesh)
        pl.render()
        time.sleep(0.1)

    time.sleep(10)
    pl.close()

if __name__ == "__main__":
    main()