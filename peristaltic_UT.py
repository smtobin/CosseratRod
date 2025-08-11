
import cosserat
import mesh
import utils
import vtk

import os
import time

import pyvista as pv
import numpy as np

SIM_PATHS = [
    "cxx/output/U_right_half_sim1",
    "cxx/output/U_left_half_sim1",
    "cxx/output/T_top_sim2",
    "cxx/output/T_bottom_sim1",
    "cxx/output/K_left_sim1",
    "cxx/output/K_bent_sim1"
]

SIM_START_AND_END_STEPS = [
    [0,225],
    [0,475],
    [0,310],
    [0,311],
    [0,311],
    [0,230]
]

SIM_STEP_OFFSET = [
    0,
    0,
    0,
    0,
    120,
    140
]

def rotZ(angle):
    rad = angle*np.pi/180
    return np.matrix(
        [[np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0,0,1]]
    )

SIM_START_ROTATION = [
    rotZ(-126),
    rotZ(170),
    rotZ(-59),
    rotZ(215),
    rotZ(180),
    rotZ(0)
]

SIM_START_POSITION = [
    np.array([0,0,0]),
    np.array([0.54, 3.66,0]),
    np.array([2.6,-3.8,0]),
    np.array([0,-5.2,0]),
    np.array([4.55,1.45, 0]),
    np.array([7.2,-3.5,0])
]

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
            if np.linalg.norm(orientation) < 1e-12:
                orientations.append(np.eye(3))
            else:
                orientations.append(utils.MatExp_so3(orientation/np.linalg.norm(orientation), np.linalg.norm(orientation)))

    return rod, actuator_pressures, positions, orientations, states
       
class RodSim:
    def __init__(self, rod, positions, orientations, states, start_pos, start_or, frame_bounds, offset):
        self.rod = rod
        self.positions = positions
        self.orientations = orientations
        self.states = states
        self.start_pos = start_pos
        self.start_or = start_or
        self.frame_bounds = frame_bounds
        self.offset = offset

def main():
    
    rod_sims = []
    for i,sim_path in enumerate(SIM_PATHS):
        (rod, actuator_pressures, positions, orientations, states) = readSimFromFolder(sim_path)
        rod_sims.append(RodSim(rod, positions, orientations, states,
                                SIM_START_POSITION[i], SIM_START_ROTATION[i], SIM_START_AND_END_STEPS[i], SIM_STEP_OFFSET[i]))

        print(f"Rod {i} final position: {rod_sims[i].positions[rod_sims[i].frame_bounds[1]]}")
        print(f"Rod {i} final orientation: {rod_sims[i].orientations[rod_sims[i].frame_bounds[1]]}")

    step_start = 20
    


    ##########################################
    # Set up plotter
    ###########################################
    pl = pv.Plotter()
    pl.camera.tight = False
    pl.enable_anti_aliasing()
    # pl.show_grid()
    # pl.camera.enable_parallel_projection()
    # pl.camera_position = 'xy'
    pl.camera_position = [1,-4,10]
    pl.camera.focal_point = [1.01, -4, 0]
    pl.camera.up = [0,1,0]
    # pl.camera.zoom(0.3)
    # pl.camera.position = [0, 0, 1]
    # pl.camera.clipping_range = (0.01, 1000.01)

    ######################
    # Pipe sim stuff
    ######################
    # pl.camera_position = [-5, 0, 0.5]
    # pl.camera.focal_point = [0, 0, 0.49]

    # cylinder = pv.Cylinder(radius=0.2, height=5, resolution=50, direction=(0,0,1))
    # cylinder = cylinder.triangulate()
    # inner_cylinder = pv.Cylinder(radius=0.12, height=5, resolution=50, direction=(0,0,1))
    # inner_cylinder = inner_cylinder.triangulate()
    # pipe = cylinder - inner_cylinder
    # half_pipe = pipe.clip('x', value=0, invert=False)
    # pl.add_mesh(half_pipe, color=[150,150,150])

    ############################################
    # Create initial meshes (that will be updated each time step)
    ############################################

    pv_meshes = []
    pv_cross_section_meshes = []

    for i,rod_sim in enumerate(rod_sims):
        rod_mesh = rod.asMesh(rod_sim.rod.n//2, rod_sim.positions[rod_sim.frame_bounds[0]], rod_sim.orientations[rod_sim.frame_bounds[0]])
        # the rod
        pv_mesh = pv.PolyData(rod_mesh.points, faces=rod_mesh.faces)
        pv_meshes.append(pv_mesh)

        pl.add_mesh(pv_meshes[i], color=MODEL_COLOR, opacity=1, specular=1.0)

        # one for each cross section
        xsections = rod.nodeCrossSectionPolyData(rod_sim.rod.n//2, rod_sim.positions[rod_sim.frame_bounds[0]], rod_sim.orientations[rod_sim.frame_bounds[0]])
        cross_section_meshes = []
        for xsection in xsections:
            pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
            cross_section_meshes.append(pv_cs_mesh)
        pv_cross_section_meshes.append(cross_section_meshes)

        # for j,_ in enumerate(pv_cross_section_meshes):
        #     pl.add_mesh(pv_cross_section_meshes[i][j], color=MODEL_COLOR, opacity=1.0, show_edges=True, edge_color='k')    

    # start plotting
    pl.show(auto_close=False, interactive_update=True, window_size=[1920,1080])
    # pl.show()

    #########################################
    # Animate through the sim states
    #########################################

    # loop through the simulation states and update the meshes
    wait_time = 1/30 # s
    for step in range(step_start,500):
        print(step)
        for i,rod_sim in enumerate(rod_sims):
            if step < rod_sim.offset:
                continue
            sim_step = step - rod_sim.offset
            if sim_step >= len(rod_sim.states) or sim_step >= rod_sim.frame_bounds[1]:
                continue
            elif sim_step == rod_sim.frame_bounds[1]-1:
                rod_sim.states[sim_step][:2*rod_sim.rod.n] = np.ones(2*rod_sim.rod.n)

            # update the rod state and get its new mesh
            cur_pos = rod_sim.start_pos + np.matmul(rod_sim.start_or, rod_sim.positions[sim_step])
            cur_or = np.matmul(rod_sim.start_or, rod_sim.orientations[sim_step])
            rod_sim.rod.Z = rod_sim.states[sim_step]
            new_rod_mesh = rod_sim.rod.asMesh(rod_sim.rod.n//2, cur_pos, cur_or)

            new_pv_mesh = pv.PolyData(new_rod_mesh.points, faces=new_rod_mesh.faces)
            pv_meshes[i].shallow_copy(new_pv_mesh)

            # update the cross section meshes
            # new_xsections = rod_sim.rod.nodeCrossSectionPolyData(rod_sim.rod.n//2, cur_pos, cur_or)
            # for j,xsection in enumerate(new_xsections):
            #     new_pv_cs_mesh = pv.PolyData(xsection.points, faces=xsection.faces)
            #     pv_cross_section_meshes[i][j].shallow_copy(new_pv_cs_mesh)

        # camera_x = min( max(step-40,0)/150, 2.5)
        camera_z = min(5+max(step-40,0)/30, 15)
        pl.camera.position = [2.5,-3.5,camera_z]
        pl.camera.focal_point = [2.5-0.01, -3.5, 0]
        pl.camera.up = [0,1,0]
        pl.reset_camera_clipping_range()

        pl.render()
        time.sleep(wait_time)

    time.sleep(10)
    pl.close()

if __name__ == "__main__":
    main()